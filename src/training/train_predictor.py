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
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)  # Suppress division by zero warnings
import torch.utils.data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import DEVICE, DRUG_EMBED_DIM
from models.pharmacodynamicPredictor import (
    PharmacodynamicPredictor, 
    LAB_BIOMARKER_FEATURES,
    PATIENT_FEATURES
)
from models.patient_generator_gan import PatientGenerator
from encoders.drugEncoder import DrugEncoder
from utils.dataset_novel_drug import NovelDrugDataset, create_novel_drug_splits
from training.losses_pharmacology import PharmacologyConstraintLoss
from data_generation.real_data_augmenter import prepare_real_data_for_training


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
# PREREQUISITE SCRIPTS
# ============================================================================

def run_prerequisite_scripts():
    """Run prerequisite scripts to regenerate data with latest improvements."""
    data_path = Path('data/cdisc/clinical_trials_responses.csv')
    
    print("\n" + "=" * 80)
    print("REGENERATING SYNTHETIC DATA")
    print("=" * 80)
    print(f"\n🔄 Regenerating {data_path} with improved constraints...")
    print("   This ensures data quality: non-negative values, realistic deltas\n")
    
    # Always regenerate to use latest improvements
    try:
        from data_generation.clinical_trials_extractor import main as extractor_main
        extractor_main()
        print("\n✓ Synthetic data regeneration completed successfully!")
    except Exception as e:
        print(f"\n❌ Failed to regenerate synthetic data: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Verify the file was created
    if not data_path.exists():
        raise FileNotFoundError(f"Expected data file was not created: {data_path}")
    
    print(f"✓ Data file ready: {data_path}")
    print(f"   File size: {data_path.stat().st_size / 1024:.1f} KB\n")


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    print("=" * 80)
    print("TRAINING FOR NOVEL DRUG GENERALIZATION")
    print("=" * 80)
    
    # Run prerequisite scripts if needed
    run_prerequisite_scripts()
    
    # Hyperparameters - SIMPLIFIED MODEL (Easier but still decent)
    BATCH_SIZE = 256  # Larger batch for more stable gradients
    EPOCHS = 200  # More epochs with early stopping
    LEARNING_RATE = 2e-4  # Higher learning rate for simpler model
    WEIGHT_DECAY = 5e-4  # Reduced weight decay
    DROPOUT = 0.2  # Lower dropout for easier learning
    LABEL_SMOOTHING = 0.05  # Reduced label smoothing
    EARLY_STOPPING_PATIENCE = 20  # More patience with larger dataset
    VAL_SPLIT = 0.2  # 20% of training data for validation
    
    # Model architecture - VERY SIMPLIFIED (Easier for larger dataset)
    HIDDEN_DIM = 64  # Further simplified from 96
    NUM_HEADS = 2  # Keep 2 heads (minimum for cross-attention)
    NUM_LAYERS = 1  # Single layer (simplest that still works)
    
    # 1. Initialize DrugEncoder (CRITICAL: Use hybrid mode for best generalization)
    print("\n[1/6] Initializing DrugEncoder (Hybrid: Transformer + RDKit)...")
    drug_encoder = DrugEncoder(
        encoder_type='hybrid',  # Best for novel drugs
        device=DEVICE,
        normalize_embeddings=True  # L2 normalize for stability
    )
    print(f"  ✓ DrugEncoder ready ({drug_encoder.num_parameters:,} params)")
    
    # 2. Load and augment real CDISC data
    print("\n[2/7] Extracting and augmenting real CDISC data...")
    real_data_augmented = None
    temp_real_path = None
    try:
        real_data_augmented = prepare_real_data_for_training(
            adsl_path='data/cdisc/adsl.csv',
            adlbc_path='data/cdisc/adlbc.csv',
            drug_smiles_path='data/mappings/drug_smiles_mapping.json',
            n_augmentations_per_sample=10,  # 10 augmentations per real sample
            noise_std=0.1,  # 10% noise
            scale_range=(0.8, 1.2),  # Scale deltas by 0.8-1.2x
            seed=42
        )
        if len(real_data_augmented) > 0:
            print(f"  ✓ Real data augmentations: {len(real_data_augmented):,} samples")
        else:
            print(f"  ⚠️  No real data augmentations generated")
            real_data_augmented = None
    except Exception as e:
        print(f"  ⚠️  Failed to load real data: {e}")
        print(f"  Continuing with synthetic data only...")
        real_data_augmented = None
    
    # 3. Load synthetic data and create drug-aware split
    print("\n[3/7] Creating drug-aware train/test split...")
    data_path = Path('data/cdisc/clinical_trials_responses.csv')
    
    if not data_path.exists():
        print(f"❌ Data not found at {data_path}")
        print("   This should not happen if prerequisites ran successfully.")
        return
    
    train_data, test_data, train_drugs, test_drugs = create_novel_drug_splits(
        data_path=str(data_path),
        drug_encoder=drug_encoder,
        test_ratio=0.2,  # 20% of DRUGS held out (not 20% of samples!)
        n_clusters=10,
        seed=42
    )
    
    print(f"\n  CRITICAL: Test drugs are UNSEEN during training!")
    print(f"  This tests generalization to new molecular structures")
    print(f"\n  Example test drugs: {test_drugs[:5]}")
    
    # 4. Combine real augmentations with synthetic data (ONLY for training)
    if real_data_augmented is not None and len(real_data_augmented) > 0:
        print("\n[4/7] Combining real augmentations with synthetic data...")
        
        # Estimate target size: balance real augmentations with synthetic
        synthetic_train_size = len(train_data)
        # Use 20-30% of training data as real augmentations
        target_real_size = int(synthetic_train_size * 0.25)
        
        # Balance real augmentations to target size
        from data_generation.real_data_augmenter import balance_augmented_data
        real_data_augmented = balance_augmented_data(
            real_data_augmented,
            target_size=min(target_real_size, len(real_data_augmented)),
            drug_balance=True,
            seed=42
        )
        
        # Save real augmentations to temporary CSV
        temp_real_path = Path('data/cdisc/real_augmentations_temp.csv')
        temp_real_path.parent.mkdir(parents=True, exist_ok=True)
        real_data_augmented.to_csv(temp_real_path, index=False)
        print(f"  ✓ Saved {len(real_data_augmented):,} real augmentations to {temp_real_path}")
        print(f"  ✓ Real augmentations will be added to training set only")
        print(f"  ✓ Synthetic data: {len(train_data):,} samples")
        print(f"  ✓ Real augmentations: {len(real_data_augmented):,} samples")
        print(f"  ✓ Combined training: {len(train_data) + len(real_data_augmented):,} samples")
    else:
        temp_real_path = None
    
    # 5. Create datasets
    print("\n[5/7] Creating datasets...")
    
    # Create synthetic training dataset
    train_dataset = NovelDrugDataset(
        data_path=str(data_path),
        drug_encoder=drug_encoder,
        split='train',
        train_drugs=train_drugs,
        test_drugs=test_drugs,
        seed=42
    )
    
    # Add real augmentations to training dataset if available
    if temp_real_path is not None and temp_real_path.exists():
        print(f"\n  Adding real augmentations to training dataset...")
        try:
            # Load real augmentations CSV
            real_df = pd.read_csv(temp_real_path)
            real_drugs = real_df['drug_name'].unique().tolist()
            print(f"    Real augmentation drugs: {real_drugs}")
            
            # For real augmentations, include ALL drugs (don't filter by train_drugs/test_drugs)
            # This ensures we use all available real data for training
            # Create a modified train_drugs list that includes real augmentation drugs
            extended_train_drugs = list(set(train_drugs + real_drugs))
            
            real_dataset = NovelDrugDataset(
                data_path=str(temp_real_path),
                drug_encoder=drug_encoder,
                split='train',  # Treat as training data
                train_drugs=extended_train_drugs,  # Include real augmentation drugs
                test_drugs=test_drugs,
                seed=42
            )
            
            # Combine datasets using ConcatDataset
            from torch.utils.data import ConcatDataset
            train_dataset = ConcatDataset([train_dataset, real_dataset])
            print(f"  ✓ Combined dataset: {len(train_dataset):,} total samples")
            print(f"    - Synthetic: {len(train_dataset.datasets[0]):,}")
            print(f"    - Real augmentations: {len(train_dataset.datasets[1]):,}")
        except Exception as e:
            print(f"  ⚠️  Failed to add real augmentations: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Continuing with synthetic data only...")
    
    test_dataset = NovelDrugDataset(
        data_path=str(data_path),
        drug_encoder=drug_encoder,
        split='test',
        train_drugs=train_drugs,
        test_drugs=test_drugs,
        seed=42
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
    
    print(f"  ✓ Train: {len(train_dataset):,} samples from {len(train_drugs)} drugs")
    print(f"  ✓ Test: {len(test_dataset):,} samples from {len(test_drugs)} NOVEL drugs")
    
    # 6. Split training data into train/validation
    print("\n[6/7] Splitting training data into train/validation...")
    from sklearn.model_selection import train_test_split
    train_indices = np.arange(len(train_dataset))
    train_idx, val_idx = train_test_split(
        train_indices, 
        test_size=VAL_SPLIT, 
        random_state=42,
        shuffle=True
    )
    
    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(train_dataset, val_idx)
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    print(f"  ✓ Train: {len(train_subset):,} samples")
    print(f"  ✓ Validation: {len(val_subset):,} samples")
    print(f"  ✓ Test (Novel Drugs): {len(test_dataset):,} samples")
    
    # 7. Initialize model with SIMPLIFIED ARCHITECTURE
    print("\n[7/8] Initializing PharmacodynamicPredictor (simplified but decent)...")
    print(f"  Architecture: hidden_dim={HIDDEN_DIM}, num_heads={NUM_HEADS}, num_layers={NUM_LAYERS}, dropout={DROPOUT}")
    predictor = PharmacodynamicPredictor(
        predictor_type='cross_attention',
        hidden_dim=HIDDEN_DIM,  # Simplified: 96 (was 128)
        num_heads=NUM_HEADS,  # Simplified: 2 (was 4)
        num_layers=NUM_LAYERS,  # Simplified: 1 (was 2)
        dropout=DROPOUT,  # Moderate dropout: 0.3 (was 0.5)
        use_constraints=False,  # Disable during training
        device=DEVICE
    )
    
    # Calculate and print model size
    total_params = sum(p.numel() for p in predictor.model.parameters())
    trainable_params = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
    print(f"  ✓ Model initialized: {total_params / 1e6:.2f}M total parameters ({trainable_params / 1e6:.2f}M trainable)")
    print(f"  ✓ Simplified architecture for better generalization")
    
    # 8. Initialize constraint loss (SIMPLIFIED weights)
    print("\n[8/8] Initializing pharmacology constraint loss (simplified weights)...")
    criterion = PharmacologyConstraintLoss(
        lambda_monotonicity=0.005,  # Simplified: minimal constraints
        lambda_mechanism=0.02,      # Simplified: let model learn naturally
        lambda_consistency=0.01,    # Simplified: minimal consistency
        lambda_bounds=0.01,         # Simplified: minimal bounds
        drug_encoder=drug_encoder  # Pass encoder to compute real mechanism signatures
    ).to(DEVICE)
    
    # Base MSE loss with label smoothing
    base_criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(
        predictor.model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY  # Increased for regularization
    )
    
    # Learning rate scheduler with warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
    
    # Warmup + cosine annealing
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # Linear warmup
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # 9. Training loop with early stopping
    print("\n[9/9] Training with early stopping...")
    print(f"  Strategy: Learn mechanisms from {len(train_drugs)} drugs")
    print(f"  Goal: Generalize to {len(test_drugs)} unseen drugs")
    print()
    
    best_val_r2 = -float('inf')
    best_val_loss = float('inf')
    best_test_r2 = -float('inf')
    best_test_loss = float('inf')
    best_val_mape = float('inf')
    best_test_mape = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Train
        predictor.model.train()
        train_loss = 0.0
        train_loss_components = {
            'base': 0.0,
            'monotonicity': 0.0,
            'mechanism': 0.0,
            'consistency': 0.0,
            'bounds': 0.0
        }
        
        for batch in train_loader:
            patient_state = batch['patient_state'].to(DEVICE)
            drug_emb = batch['drug_emb'].to(DEVICE)
            lab_delta_true = batch['lab_delta'].to(DEVICE)
            
            optimizer.zero_grad()
            lab_delta_pred, _ = predictor.model(patient_state, drug_emb, return_attention=False)
            
            # Base loss with label smoothing
            base_loss = base_criterion(lab_delta_pred, lab_delta_true)
            
            # Apply label smoothing: mix true labels with uniform distribution
            if LABEL_SMOOTHING > 0:
                # Create smoothed targets: (1 - alpha) * true + alpha * uniform
                # Use zero-centered noise instead of batch statistics for better regularization
                noise = torch.randn_like(lab_delta_true) * lab_delta_true.std() * 0.3
                smoothed_target = (1 - LABEL_SMOOTHING) * lab_delta_true + LABEL_SMOOTHING * noise
                base_loss = base_criterion(lab_delta_pred, smoothed_target)
            
            # Constraint loss
            constraint_loss, components = criterion(
                lab_delta_pred, lab_delta_true,
                drug_emb, patient_state,
                return_components=True
            )
            
            # Combined loss
            loss = base_loss + constraint_loss
            
            loss.backward()
            # Very aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), 0.5)  # Very tight clipping
            optimizer.step()
            
            train_loss += loss.item()
            for key in train_loss_components:
                train_loss_components[key] += components[key]
        
        train_loss /= len(train_loader)
        for key in train_loss_components:
            train_loss_components[key] /= len(train_loader)
        
        # Evaluate on VALIDATION set (for early stopping)
        predictor.model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                patient_state = batch['patient_state'].to(DEVICE)
                drug_emb = batch['drug_emb'].to(DEVICE)
                lab_delta_true = batch['lab_delta'].to(DEVICE)
                
                lab_delta_pred, _ = predictor.model(patient_state, drug_emb, return_attention=False)
                loss = nn.MSELoss()(lab_delta_pred, lab_delta_true)
                val_loss += loss.item()
                
                val_preds.append(lab_delta_pred.cpu().numpy())
                val_targets.append(lab_delta_true.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_r2 = r2_score(val_targets, val_preds)
        
        val_mae = mean_absolute_error(val_targets, val_preds)
        
        # Calculate MAPE (Mean Absolute Percentage Error) - perfect for regression
        # MAPE = mean(|actual - predicted| / |actual|) * 100
        # Lower is better, shows average error as percentage
        val_abs_targets = np.abs(val_targets)
        val_errors = np.abs(val_targets - val_preds)
        # Avoid division by zero: use small epsilon for near-zero targets
        val_mape = float(np.mean(np.where(val_abs_targets > 1e-6, 
                                          val_errors / (val_abs_targets + 1e-6), 
                                          val_errors)) * 100)
        
        # Evaluate on NOVEL drugs (test set)
        test_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                patient_state = batch['patient_state'].to(DEVICE)
                drug_emb = batch['drug_emb'].to(DEVICE)
                lab_delta_true = batch['lab_delta'].to(DEVICE)
                
                lab_delta_pred, _ = predictor.model(patient_state, drug_emb, return_attention=False)
                loss = nn.MSELoss()(lab_delta_pred, lab_delta_true)
                test_loss += loss.item()
                
                all_preds.append(lab_delta_pred.cpu().numpy())
                all_targets.append(lab_delta_true.cpu().numpy())
        
        test_loss /= len(test_loader)
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        test_r2 = r2_score(all_targets, all_preds)
        test_mae = mean_absolute_error(all_targets, all_preds)
        
        # Calculate MAPE (Mean Absolute Percentage Error) - perfect for regression
        # MAPE = mean(|actual - predicted| / |actual|) * 100
        # Lower is better, shows average error as percentage
        test_abs_targets = np.abs(all_targets)
        test_errors = np.abs(all_targets - all_preds)
        # Avoid division by zero: use small epsilon for near-zero targets
        test_mape = float(np.mean(np.where(test_abs_targets > 1e-6, 
                                           test_errors / (test_abs_targets + 1e-6), 
                                           test_errors)) * 100)
        
        # Update scheduler (per epoch)
        scheduler.step()
        
        # Also compute per-feature R² to see which biomarkers are learning
        if epoch % 20 == 0:
            per_feature_r2 = []
            for i in range(all_targets.shape[1]):
                try:
                    feat_r2 = r2_score(all_targets[:, i], all_preds[:, i])
                    per_feature_r2.append((LAB_BIOMARKER_FEATURES[i], feat_r2))
                except:
                    per_feature_r2.append((LAB_BIOMARKER_FEATURES[i], -999))
            
            # Show top 3 and bottom 3 features
            per_feature_r2.sort(key=lambda x: x[1], reverse=True)
            print(f"    Top 3 features: {per_feature_r2[:3]}")
            print(f"    Bottom 3 features: {per_feature_r2[-3:]}")
        
        # Early stopping and model saving based on validation MAPE (best metric for regression)
        if val_mape < best_val_mape:
            best_val_loss = val_loss
            best_val_r2 = val_r2
            best_test_r2 = test_r2
            best_val_mape = val_mape
            best_test_mape = test_mape
            best_test_loss = test_loss
            best_epoch = epoch
            patience_counter = 0
            predictor.save('models/pharmacodynamic_predictor/predictor_novel_drug_best.pt')
            print(f"  ✓ New best model! Val MAPE: {val_mape:.1f}% (saved)")
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else LEARNING_RATE
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} (MAPE: {val_mape:.1f}%) | "
                  f"Test: {test_loss:.4f} (MAPE: {test_mape:.1f}%) | "
                  f"LR: {current_lr:.2e}")
            print(f"  Loss components: "
                  f"Base={train_loss_components['base']:.3f}, "
                  f"Mechanism={train_loss_components['mechanism']:.3f}, "
                  f"Consistency={train_loss_components['consistency']:.3f}")
            if patience_counter > 0:
                print(f"  ⚠️  No improvement for {patience_counter} epochs (patience: {EARLY_STOPPING_PATIENCE})")
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n🛑 Early stopping triggered at epoch {epoch}")
            print(f"   Best validation MAPE: {best_val_mape:.1f}% (epoch {best_epoch})")
            break
    
    total_time = time.time() - start_time
    
    print(f"\n✓ Training complete in {total_time/60:.1f} minutes!")
    print(f"✓ Best validation MAPE: {best_val_mape:.1f}% (epoch {best_epoch})")
    print(f"✓ Best test MAPE on novel drugs: {best_test_mape:.1f}% (epoch {best_epoch})")
    print(f"  (R²: Val={best_val_r2:.4f}, Test={best_test_r2:.4f})")
    print(f"\nTarget: MAPE < 20% (average error < 20% of true values)")
    
    if best_test_r2 > 0.64:
        print("🎉 SUCCESS! Model generalizes well to novel drugs!")
    elif best_test_r2 > 0.4:
        print("⚠️  Moderate generalization. Consider:")
        print("   - More training data")
        print("   - Stronger mechanism constraints")
        print("   - Data augmentation (SMILES randomization)")
    else:
        print("❌ Poor generalization. Need:")
        print("   - More diverse training drugs")
        print("   - Better molecular descriptors")
        print("   - Meta-learning approach (MAML)")
    
    print(f"\nBest model saved to: models/pharmacodynamic_predictor/predictor_novel_drug_best.pt")
    
    # Cleanup temporary file
    if temp_real_path is not None and temp_real_path.exists():
        temp_real_path.unlink()
        print(f"✓ Cleaned up temporary augmentation file")
    
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
