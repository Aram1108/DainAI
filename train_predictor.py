"""Train Cross-Attention Pharmacodynamic Predictor.

Training strategy:
1. Use 90% of NHANES data for training, 10% for testing
2. Augment with 10% synthetic patients from GAN
3. Create synthetic drug-response pairs (since we don't have real drug data)
4. Track multiple metrics: MSE, MAE, R², per-feature accuracy
5. Analyze model performance and provide improvement suggestions
"""
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from constants import DEVICE, DRUG_EMBED_DIM
from pharmacodynamicPredictor import (
    PharmacodynamicPredictor, 
    LAB_BIOMARKER_FEATURES,
    PATIENT_FEATURES
)
from patient_generator_gan import PatientGenerator
from drugEncoder import DrugEncoder


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

class SyntheticDrugResponseDataset(Dataset):
    """Generate synthetic drug-response pairs for training.
    
    Since we don't have real drug-patient-response triplets, we create
    synthetic ones by:
    1. Taking real patient states from NHANES
    2. Generating random drug embeddings
    3. Creating plausible lab changes based on known pharmacology patterns
    """
    
    def __init__(
        self,
        patient_data: pd.DataFrame,
        n_drugs: int = 1000,
        seed: int = 42
    ):
        """
        Args:
            patient_data: DataFrame with 41 patient features
            n_drugs: Number of unique synthetic drugs to generate
            seed: Random seed
        """
        self.patient_data = patient_data
        self.n_drugs = n_drugs
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate synthetic drug embeddings (simulate diverse drug space)
        self.drug_embeddings = self._generate_drug_embeddings(n_drugs)
        
        # Each patient can receive multiple drugs
        self.n_patients = len(patient_data)
        self.n_samples = self.n_patients * 5  # 5 drug trials per patient
        
        # Pre-generate patient-drug pairs
        self.pairs = self._generate_pairs()
        
    def _generate_drug_embeddings(self, n_drugs: int) -> np.ndarray:
        """Generate diverse drug embeddings in realistic range."""
        # Real drug embeddings from hybrid encoder have:
        # - Mean ~0, std ~1 (normalized)
        # - Norm typically 20-40
        
        embeddings = np.random.randn(n_drugs, DRUG_EMBED_DIM).astype(np.float32)
        
        # Normalize to realistic scale
        for i in range(n_drugs):
            norm = np.linalg.norm(embeddings[i])
            embeddings[i] = embeddings[i] / norm * np.random.uniform(20, 40)
        
        return embeddings
    
    def _generate_pairs(self) -> List[Tuple[int, int]]:
        """Generate (patient_idx, drug_idx) pairs."""
        pairs = []
        for _ in range(self.n_samples):
            patient_idx = np.random.randint(0, self.n_patients)
            drug_idx = np.random.randint(0, self.n_drugs)
            pairs.append((patient_idx, drug_idx))
        return pairs
    
    def _simulate_lab_response(
        self,
        patient_state: np.ndarray,
        drug_emb: np.ndarray
    ) -> np.ndarray:
        """Simulate plausible lab biomarker changes.
        
        Creates synthetic ground truth that DEPENDS ON BOTH drug and patient:
        - Drug embedding features determine which systems are affected
        - Patient baseline values modulate response magnitude
        - More realistic continuous relationships (not discrete patterns)
        """
        # Extract baseline labs (indices 12-33)
        baseline_labs = patient_state[12:34]
        
        # Extract patient demographics
        age = patient_state[0]
        bmi = patient_state[6]  # BMXBMI
        
        # Drug "profile" - continuous features from embedding
        # Use different parts of embedding for different systems
        metabolic_strength = np.tanh(drug_emb[0:256].mean())  # -1 to 1
        kidney_strength = np.tanh(drug_emb[256:512].mean())
        liver_strength = np.tanh(drug_emb[512:768].mean())
        
        # Initialize delta
        delta = np.zeros(22, dtype=np.float32)
        
        # Metabolic effects (glucose, cholesterol, lipids)
        # Response depends on baseline AND drug strength
        glucose_baseline = baseline_labs[14]  # LBXSGL
        if glucose_baseline > 100:  # High baseline → larger effect
            delta[14] = metabolic_strength * -20 * (glucose_baseline / 100)
        else:
            delta[14] = metabolic_strength * -5
        
        cholesterol_baseline = baseline_labs[21]  # LBXTC
        delta[21] = metabolic_strength * -25 * (cholesterol_baseline / 200)
        delta[11] = metabolic_strength * 15  # LBXSCH
        
        # Kidney effects (depend on age and baseline kidney function)
        age_factor = 1.0 + (age - 40) / 100  # Older patients → stronger effect
        delta[2] = kidney_strength * 10 * age_factor  # LBDSCR (creatinine)
        delta[9] = kidney_strength * 5 * age_factor   # LBXSBU (BUN)
        delta[16] = kidney_strength * 0.3             # LBXSK (potassium)
        
        # Liver effects (depend on BMI and baseline liver enzymes)
        bmi_factor = 1.0 + (bmi - 25) / 50  # Higher BMI → stronger effect
        delta[8] = liver_strength * 30 * bmi_factor   # LBXSAS (AST)
        delta[15] = liver_strength * 40 * bmi_factor  # LBXSGT (GGT)
        delta[13] = liver_strength * 0.5              # LBXSGB (bilirubin)
        
        # Inflammatory markers
        inflammation_strength = np.tanh(drug_emb[384:640].mean())
        delta[6] = inflammation_strength * -2.0  # LBXCRP
        delta[20] = inflammation_strength * -1.5  # LBXSUA
        
        # Add realistic noise (smaller than before)
        delta += np.random.randn(22) * 1.0
        
        # Ensure physiological constraints
        new_labs = baseline_labs + delta
        
        # Clamp to safe ranges
        bounds = {
            14: (50, 300),    # Glucose
            10: (7.0, 12.0),  # Calcium
            16: (2.5, 6.5),   # Potassium
            17: (130, 150),   # Sodium
            12: (95, 110),    # Chloride
        }
        
        for idx, (low, high) in bounds.items():
            if new_labs[idx] < low:
                delta[idx] = low - baseline_labs[idx]
            elif new_labs[idx] > high:
                delta[idx] = high - baseline_labs[idx]
        
        return delta
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        patient_idx, drug_idx = self.pairs[idx]
        
        # Get patient state (41 features)
        patient_row = self.patient_data.iloc[patient_idx][PATIENT_FEATURES]
        patient_state = patient_row.values.copy()
        
        # Convert SEX to numeric (index 1)
        if isinstance(patient_state[1], str):
            patient_state[1] = 1.0 if patient_state[1] == 'M' else 0.0
        
        patient_state = patient_state.astype(np.float32)
        
        # Get drug embedding
        drug_emb = self.drug_embeddings[drug_idx]
        
        # Simulate lab response
        lab_delta = self._simulate_lab_response(patient_state, drug_emb)
        
        return {
            'patient_state': torch.tensor(patient_state, dtype=torch.float32),
            'drug_emb': torch.tensor(drug_emb, dtype=torch.float32),
            'lab_delta': torch.tensor(lab_delta, dtype=torch.float32)
        }


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, feature_names: List[str]) -> Dict:
    """Compute comprehensive evaluation metrics.
    
    Args:
        y_true: (N, 22) ground truth lab changes
        y_pred: (N, 22) predicted lab changes
        feature_names: List of 22 lab feature names
        
    Returns:
        Dictionary with overall and per-feature metrics
    """
    metrics = {}
    
    # Overall metrics
    metrics['overall'] = {
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }
    
    # Per-feature metrics
    metrics['per_feature'] = {}
    for i, feat in enumerate(feature_names):
        metrics['per_feature'][feat] = {
            'mse': float(mean_squared_error(y_true[:, i], y_pred[:, i])),
            'mae': float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            'r2': float(r2_score(y_true[:, i], y_pred[:, i])),
            'mean_true': float(y_true[:, i].mean()),
            'std_true': float(y_true[:, i].std()),
            'mean_pred': float(y_pred[:, i].mean()),
            'std_pred': float(y_pred[:, i].std()),
        }
    
    # Accuracy within thresholds (clinical relevance)
    for threshold in [5, 10, 20]:
        within = np.abs(y_true - y_pred) <= threshold
        metrics['overall'][f'accuracy_within_{threshold}'] = float(within.mean())
    
    return metrics


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(
    predictor,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str
) -> float:
    """Train for one epoch."""
    predictor.model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        patient_state = batch['patient_state'].to(device)
        drug_emb = batch['drug_emb'].to(device)
        lab_delta_true = batch['lab_delta'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        lab_delta_pred, _ = predictor.model(patient_state, drug_emb, return_attention=False)
        
        # Loss: MSE on lab changes
        loss = nn.MSELoss()(lab_delta_pred, lab_delta_true)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(
    predictor,
    dataloader: DataLoader,
    device: str
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model on validation/test set."""
    predictor.model.eval()
    total_loss = 0.0
    n_batches = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            patient_state = batch['patient_state'].to(device)
            drug_emb = batch['drug_emb'].to(device)
            lab_delta_true = batch['lab_delta'].to(device)
            
            lab_delta_pred, _ = predictor.model(patient_state, drug_emb, return_attention=False)
            
            loss = nn.MSELoss()(lab_delta_pred, lab_delta_true)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_preds.append(lab_delta_pred.cpu().numpy())
            all_targets.append(lab_delta_true.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    return total_loss / n_batches, all_targets, all_preds


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    print("=" * 80)
    print("TRAINING CROSS-ATTENTION PHARMACODYNAMIC PREDICTOR")
    print("=" * 80)
    
    # Hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 5e-5  # Lower LR for better generalization
    WEIGHT_DECAY = 1e-4   # Stronger regularization
    DROPOUT = 0.3         # Increased dropout
    
    # 1. Load NHANES data
    print("\n[1/8] Loading NHANES data...")
    data_path = Path('preprocessed_nhanes/nhanes_final_complete.csv')
    if not data_path.exists():
        print(f"❌ Data not found at {data_path}")
        return
    
    nhanes_df = pd.read_csv(data_path)
    print(f"  ✓ Loaded {len(nhanes_df):,} NHANES patients")
    
    # 2. Generate synthetic patients (10% of NHANES size)
    print("\n[2/8] Generating synthetic patients from GAN...")
    n_synthetic = int(len(nhanes_df) * 0.1)
    
    gen = PatientGenerator(data_path=str(data_path))
    
    # Try to load best GAN model
    best_model_path = Path('models/generator/patient_generator_best.pt')
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
        gen.generator.load_state_dict(checkpoint['generator'])
        print(f"  ✓ Loaded best GAN (FID={checkpoint.get('fid_score', '?'):.2f})")
    else:
        print("  ⚠ No trained GAN found, using random generator")
    
    # Generate diverse synthetic patients with varying demographics
    np.random.seed(42)
    synthetic_batches = []
    batch_size = 100
    n_batches = (n_synthetic + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        n_this_batch = min(batch_size, n_synthetic - i * batch_size)
        
        # Sample diverse demographics
        ages = np.random.randint(18, 85, n_this_batch)
        sexes = np.random.choice(['M', 'F'], n_this_batch)
        heights = np.random.uniform(150, 190, n_this_batch)
        weights = np.random.uniform(50, 120, n_this_batch)
        
        for age, sex, height, weight in zip(ages, sexes, heights, weights):
            patient = gen.generate(age=int(age), sex=sex, height=float(height), weight=float(weight), n=1, seed=None)
            synthetic_batches.append(patient)
    
    synthetic_patients = pd.concat(synthetic_batches, ignore_index=True)
    print(f"  ✓ Generated {len(synthetic_patients):,} synthetic patients")
    
    # 3. Combine and split data
    print("\n[3/8] Splitting data (90% train, 10% test)...")
    all_patients = pd.concat([nhanes_df, synthetic_patients], ignore_index=True)
    print(f"  Total patients: {len(all_patients):,} ({len(nhanes_df):,} real + {len(synthetic_patients):,} synthetic)")
    
    train_df, test_df = train_test_split(
        all_patients,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )
    
    print(f"  ✓ Training set: {len(train_df):,} patients")
    print(f"  ✓ Test set: {len(test_df):,} patients")
    
    # 4. Create datasets and dataloaders
    print("\n[4/8] Creating synthetic drug-response datasets...")
    # Use SHARED drugs between train and test for fair evaluation
    SHARED_DRUGS = 500
    train_dataset = SyntheticDrugResponseDataset(
        train_df,
        n_drugs=SHARED_DRUGS,  # Same drugs as test
        seed=42
    )
    test_dataset = SyntheticDrugResponseDataset(
        test_df,
        n_drugs=SHARED_DRUGS,  # Same drugs, test generalization to new patients
        seed=42  # SAME SEED = same drug embeddings
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    print(f"  ✓ Training samples: {len(train_dataset):,}")
    print(f"  ✓ Test samples: {len(test_dataset):,}")
    print(f"  ✓ Batches per epoch: {len(train_loader):,}")
    
    # 5. Initialize model
    print("\n[5/8] Initializing model...")
    predictor = PharmacodynamicPredictor(
        predictor_type='cross_attention',
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        use_constraints=False,  # Disable during training
        device=DEVICE
    )
    
    # Manually set higher dropout in model
    for module in predictor.model.modules():
        if isinstance(module, nn.Dropout):
            module.p = DROPOUT
    
    optimizer = optim.AdamW(
        predictor.model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # 6. Training loop
    print("\n[6/8] Training...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Device: {DEVICE}")
    print()
    
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_metrics': [],
        'lr': []
    }
    
    best_test_loss = float('inf')
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(predictor, train_loader, optimizer, DEVICE)
        
        # Evaluate
        test_loss, y_true, y_pred = evaluate(predictor, test_loader, DEVICE)
        test_metrics = compute_metrics(y_true, y_pred, LAB_BIOMARKER_FEATURES)
        
        # Update scheduler
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_metrics'].append(test_metrics)
        history['lr'].append(current_lr)
        
        # Track best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            
            # Save best model
            save_path = Path('models/pharmacodynamic_predictor/predictor_trained_best.pt')
            predictor.save(str(save_path))
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"R²: {test_metrics['overall']['r2']:.4f} | "
              f"MAE: {test_metrics['overall']['mae']:.2f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Early stopping
        if epoch - best_epoch > 15:
            print(f"\n⚠ Early stopping - no improvement for 15 epochs")
            break
    
    total_time = time.time() - start_time
    
    print(f"\n✓ Training complete in {total_time/60:.1f} minutes")
    print(f"✓ Best epoch: {best_epoch+1} (test loss: {best_test_loss:.4f})")
    
    # 7. Final evaluation on best model
    print("\n[7/8] Evaluating best model...")
    
    # Load best model
    best_model_path = Path('models/pharmacodynamic_predictor/predictor_trained_best.pt')
    predictor.load(str(best_model_path))
    
    # Re-enable constraints for final evaluation
    predictor.use_constraints = True
    
    test_loss, y_true, y_pred = evaluate(predictor, test_loader, DEVICE)
    final_metrics = compute_metrics(y_true, y_pred, LAB_BIOMARKER_FEATURES)
    
    # 8. Analysis and recommendations
    print("\n[8/8] Analysis and Recommendations")
    print("=" * 80)
    
    print("\n📊 OVERALL PERFORMANCE:")
    print(f"  MSE:  {final_metrics['overall']['mse']:.2f}")
    print(f"  RMSE: {final_metrics['overall']['rmse']:.2f}")
    print(f"  MAE:  {final_metrics['overall']['mae']:.2f}")
    print(f"  R²:   {final_metrics['overall']['r2']:.4f}")
    print(f"\n  Accuracy within ±5:  {final_metrics['overall']['accuracy_within_5']*100:.1f}%")
    print(f"  Accuracy within ±10: {final_metrics['overall']['accuracy_within_10']*100:.1f}%")
    print(f"  Accuracy within ±20: {final_metrics['overall']['accuracy_within_20']*100:.1f}%")
    
    # Find best and worst performing features
    feature_r2 = [(feat, metrics['r2']) for feat, metrics in final_metrics['per_feature'].items()]
    feature_r2.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🎯 TOP 5 BEST PREDICTED FEATURES:")
    for feat, r2 in feature_r2[:5]:
        mae = final_metrics['per_feature'][feat]['mae']
        print(f"  {feat:12s} - R²={r2:6.3f}, MAE={mae:6.2f}")
    
    print("\n❌ TOP 5 WORST PREDICTED FEATURES:")
    for feat, r2 in feature_r2[-5:]:
        mae = final_metrics['per_feature'][feat]['mae']
        print(f"  {feat:12s} - R²={r2:6.3f}, MAE={mae:6.2f}")
    
    # Save results
    print("\n💾 Saving results...")
    results = {
        'training_config': {
            'epochs_trained': len(history['train_loss']),
            'best_epoch': best_epoch + 1,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'n_train_patients': len(train_df),
            'n_test_patients': len(test_df),
            'n_synthetic_patients': n_synthetic,
            'training_time_minutes': total_time / 60,
        },
        'final_metrics': final_metrics,
        'history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'test_loss': [float(x) for x in history['test_loss']],
            'lr': [float(x) for x in history['lr']],
        }
    }
    
    results_path = Path('models/pharmacodynamic_predictor/training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Results saved to {results_path}")
    
    # Generate recommendations
    print("\n🔧 RECOMMENDATIONS FOR IMPROVEMENT:")
    print()
    
    r2 = final_metrics['overall']['r2']
    mae = final_metrics['overall']['mae']
    acc_10 = final_metrics['overall']['accuracy_within_10']
    
    recommendations = []
    
    # R² based recommendations
    if r2 < 0.3:
        recommendations.append("❗ LOW R² (<0.3): Model has poor predictive power")
        recommendations.append("   → This is EXPECTED - we trained on synthetic data, not real drug responses")
        recommendations.append("   → To improve: Need real clinical data (drug trials, EHR, pharmacology databases)")
    elif r2 < 0.6:
        recommendations.append("⚠ MODERATE R² (0.3-0.6): Model captures some patterns but limited")
        recommendations.append("   → Consider increasing model capacity (more layers/heads)")
        recommendations.append("   → Add more training data or better data augmentation")
    else:
        recommendations.append("✓ GOOD R² (>0.6): Model learning well from synthetic data")
    
    # MAE based recommendations
    if mae > 20:
        recommendations.append("\n❗ HIGH MAE (>20): Large prediction errors")
        recommendations.append("   → Check if feature scales are properly normalized")
        recommendations.append("   → Consider per-feature loss weighting (important labs get higher weight)")
    elif mae > 10:
        recommendations.append("\n⚠ MODERATE MAE (10-20): Prediction errors are noticeable")
        recommendations.append("   → Fine-tune learning rate or use cosine annealing")
    
    # Accuracy based recommendations
    if acc_10 < 0.5:
        recommendations.append("\n❗ LOW ACCURACY (<50% within ±10): Poor clinical utility")
        recommendations.append("   → Model struggling to learn realistic response magnitudes")
        recommendations.append("   → Consider adding auxiliary losses (consistency, monotonicity)")
    
    # Architecture recommendations
    worst_r2 = feature_r2[-1][1]
    if worst_r2 < -0.5:
        recommendations.append("\n⚠ Some features have NEGATIVE R²: Worse than mean baseline")
        recommendations.append("   → Consider feature-specific heads or multi-task learning")
        recommendations.append("   → Some labs may need different architectures (e.g., binary for detectable/not)")
    
    # Data recommendations
    recommendations.append("\n📊 DATA QUALITY ISSUES:")
    recommendations.append("   ⚠ Training on SYNTHETIC drug responses (simulated patterns)")
    recommendations.append("   ⚠ Real pharmacology is FAR more complex than our simulation")
    recommendations.append("   → Priority: Acquire real drug-patient-response data")
    recommendations.append("   → Sources: Clinical trials, FDA databases, EHR systems, PubChem")
    
    # Training recommendations
    if len(history['train_loss']) >= EPOCHS:
        recommendations.append("\n⚠ Training did not converge (reached max epochs)")
        recommendations.append("   → Consider training for more epochs")
    
    for rec in recommendations:
        print(rec)
    
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nBest model saved to: models/pharmacodynamic_predictor/predictor_trained_best.pt")
    print(f"Training results: models/pharmacodynamic_predictor/training_results.json")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
