"""Training script for Time Series Predictor.

Converts static drug response predictions into temporal trajectories.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import json
from typing import Dict, List

from models.time_series_predictor import TimeSeriesPredictor, TimeSeriesConfig
from data_generation.trajectory_generator import generate_training_data
from utils.constants import DEVICE


class TimeSeriesDataset(Dataset):
    """Dataset for time series prediction."""
    
    def __init__(
        self,
        trajectory_data: pd.DataFrame,
        drug_vocab: Dict[str, int],
        metric_names: List[str],
        patient_feature_names: List[str],
        timepoints: List[int]
    ):
        """
        Args:
            trajectory_data: DataFrame from generate_training_data()
            drug_vocab: Dictionary mapping drug names to IDs
            metric_names: List of metric names
            patient_feature_names: List of patient feature names
            timepoints: List of timepoints (days)
        """
        self.data = trajectory_data
        self.drug_vocab = drug_vocab
        self.metric_names = metric_names
        self.patient_feature_names = patient_feature_names
        self.timepoints = timepoints
        
        # Get unique patients and their features
        self.patient_features = self._extract_patient_features()
        
    def _extract_patient_features(self) -> pd.DataFrame:
        """Extract patient features from trajectory data."""
        # Group by patient_id and get first row (patient features are same across metrics)
        patient_df = self.data.groupby('patient_id').first().reset_index()
        return patient_df
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        patient_id = row['patient_id']
        
        # Get patient features
        patient_row = self.patient_features[self.patient_features['patient_id'] == patient_id].iloc[0]
        patient_features = [
            patient_row.get('age', 50),
            1.0 if patient_row.get('sex') == 'M' else 0.0,
            patient_row.get('bmi', 25.0),
            patient_row.get('adherence', 1.0),
            patient_row.get('days_on_drug', 90)
        ]
        
        # Drug ID
        drug_name = row['drug_name']
        drug_id = self.drug_vocab.get(drug_name, 0)
        
        # Baseline and final delta for THIS metric
        baseline = row['baseline']
        final_delta = row['final_delta']
        metric_name = row['metric_name']
        
        # Find metric index
        metric_idx = self.metric_names.index(metric_name) if metric_name in self.metric_names else 0
        
        # Create baselines and final_deltas arrays for ALL metrics
        # (model expects all metrics, but we only have data for one)
        baselines = [0.0] * len(self.metric_names)
        final_deltas = [0.0] * len(self.metric_names)
        baselines[metric_idx] = baseline
        final_deltas[metric_idx] = final_delta
        
        # Target trajectory (delta at each timepoint for THIS metric)
        target = [row[f'day_{day}'] for day in self.timepoints]
        
        return {
            'patient_features': torch.tensor(patient_features, dtype=torch.float32),
            'drug_id': torch.tensor(drug_id, dtype=torch.long),
            'baselines': torch.tensor(baselines, dtype=torch.float32),  # All metrics
            'final_deltas': torch.tensor(final_deltas, dtype=torch.float32),  # All metrics
            'target': torch.tensor(target, dtype=torch.float32),
            'metric_idx': metric_idx  # Which metric this sample is for
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        patient_features = batch['patient_features'].to(device)
        drug_ids = batch['drug_id'].to(device)
        baselines = batch['baselines'].to(device)  # (batch, num_metrics)
        final_deltas = batch['final_deltas'].to(device)  # (batch, num_metrics)
        targets = batch['target'].to(device)  # (batch, num_timepoints)
        metric_indices = batch['metric_idx'].to(device)  # (batch,) - which metric each sample is for
        
        # Forward pass
        predicted, _ = model(patient_features, drug_ids, baselines, final_deltas, return_uncertainty=False)
        # predicted shape: (batch, num_timepoints, num_metrics)
        
        # Extract predictions for the correct metric for each sample
        # predicted[:, :, metric_indices] doesn't work, need to use gather
        batch_size = predicted.size(0)
        num_timepoints = predicted.size(1)
        metric_indices_expanded = metric_indices.unsqueeze(0).unsqueeze(2).expand(num_timepoints, batch_size, 1)
        metric_indices_expanded = metric_indices_expanded.permute(1, 0, 2)  # (batch, num_timepoints, 1)
        predicted_selected = predicted.gather(2, metric_indices_expanded).squeeze(2)  # (batch, num_timepoints)
        
        # Compute loss
        loss = criterion(predicted_selected, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            patient_features = batch['patient_features'].to(device)
            drug_ids = batch['drug_id'].to(device)
            baselines = batch['baselines'].to(device)  # (batch, num_metrics)
            final_deltas = batch['final_deltas'].to(device)  # (batch, num_metrics)
            targets = batch['target'].to(device)
            metric_indices = batch['metric_idx'].to(device)
            
            # Forward pass
            predicted, _ = model(patient_features, drug_ids, baselines, final_deltas, return_uncertainty=False)
            # predicted shape: (batch, num_timepoints, num_metrics)
            
            # Extract predictions for the correct metric
            batch_size = predicted.size(0)
            num_timepoints = predicted.size(1)
            metric_indices_expanded = metric_indices.unsqueeze(0).unsqueeze(2).expand(num_timepoints, batch_size, 1)
            metric_indices_expanded = metric_indices_expanded.permute(1, 0, 2)
            predicted_selected = predicted.gather(2, metric_indices_expanded).squeeze(2)  # (batch, num_timepoints)
            
            # Compute loss
            loss = criterion(predicted_selected, targets)
            total_loss += loss.item()
            
            # Store predictions
            all_preds.append(predicted_selected.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            n_batches += 1
    
    # Compute metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    mse = mean_squared_error(all_targets.flatten(), all_preds.flatten())
    mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
    
    return {
        'loss': total_loss / n_batches,
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse)
    }


def main():
    """Main training function."""
    print("=" * 60)
    print("Time Series Predictor Training")
    print("=" * 60)
    
    # Configuration
    config = TimeSeriesConfig(
        timepoints=[10, 20, 30, 60, 90, 180],
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1
    )
    
    # Load predictor data
    print("\n[1/5] Loading predictor data...")
    predictor_data_path = Path(__file__).parent.parent.parent / 'data' / 'cdisc' / 'clinical_trial_50k.csv'
    
    if not predictor_data_path.exists():
        print(f"ERROR: Predictor data not found at {predictor_data_path}")
        return
    
    predictor_data = pd.read_csv(predictor_data_path)
    print(f"  ✓ Loaded {len(predictor_data)} patient records")
    
    # Auto-detect all available metrics from delta columns
    delta_cols = [col for col in predictor_data.columns if col.endswith('_delta')]
    metric_names = sorted([col.replace('_delta', '') for col in delta_cols])
    print(f"  ✓ Found {len(metric_names)} metrics: {', '.join(metric_names)}")
    
    # Generate synthetic training data
    print("\n[2/5] Generating synthetic trajectories...")
    trajectory_data = generate_training_data(
        predictor_data,
        timepoints=config.timepoints,
        metric_names=metric_names
    )
    print(f"  ✓ Generated {len(trajectory_data)} trajectory samples")
    
    # Create drug vocabulary
    all_drugs = trajectory_data['drug_name'].unique()
    drug_vocab = {drug: idx for idx, drug in enumerate(sorted(all_drugs))}
    print(f"  ✓ Found {len(drug_vocab)} unique drugs")
    
    # Patient feature names
    patient_feature_names = ['age', 'sex', 'bmi', 'adherence', 'days_on_drug']
    
    # Split data
    print("\n[3/5] Splitting data...")
    train_data, val_data = train_test_split(
        trajectory_data,
        test_size=0.2,
        random_state=42,
        stratify=trajectory_data['drug_name']  # Stratify by drug
    )
    print(f"  ✓ Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(
        train_data, drug_vocab, metric_names, patient_feature_names, config.timepoints
    )
    val_dataset = TimeSeriesDataset(
        val_data, drug_vocab, metric_names, patient_feature_names, config.timepoints
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Initialize model
    print("\n[4/5] Initializing model...")
    predictor = TimeSeriesPredictor(
        drug_vocab=drug_vocab,
        metric_names=metric_names,
        patient_feature_names=patient_feature_names,
        config=config,
        device=DEVICE
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in predictor.model.parameters())
    print(f"  ✓ Model initialized ({n_params:,} parameters)")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(predictor.model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print("\n[5/5] Training model...")
    n_epochs = 50
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        
        # Train
        train_loss = train_epoch(
            predictor.model, train_loader, optimizer, criterion, DEVICE
        )
        train_losses.append(train_loss)
        
        # Validate
        val_metrics = validate(predictor.model, val_loader, criterion, DEVICE)
        val_losses.append(val_metrics['loss'])
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            model_path = Path(__file__).parent.parent.parent / 'models' / 'time_series_predictor.pt'
            model_path.parent.mkdir(exist_ok=True)
            predictor.save(str(model_path))
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
    
    # Plot training curves
    print("\n[6/6] Plotting training curves...")
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    
    plot_path = Path(__file__).parent.parent.parent / 'results' / 'time_series_training.png'
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path)
    print(f"  ✓ Saved plot to {plot_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

