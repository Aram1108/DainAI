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

def verify_data_file():
    """Verify that the main data file exists."""
    data_path = Path('data/cdisc/clinical_trial_50k.csv')
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {data_path}\n"
            "Please ensure clinical_trial_50k.csv is available."
        )
    
    print(f"✓ Data file ready: {data_path}")
    print(f"   File size: {data_path.stat().st_size / (1024*1024):.1f} MB\n")


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    print("=" * 80)
    print("TRAINING FOR NOVEL DRUG GENERALIZATION")
    print("=" * 80)
    
    # Verify data file exists
    verify_data_file()
    
    # Hyperparameters - POWERFUL MODEL with strong regularization
    BATCH_SIZE = 128  # Moderate batch size for better gradient estimates
    EPOCHS = 200  # More epochs with early stopping
    LEARNING_RATE = 8e-5  # Slightly lower LR for larger model stability
    WEIGHT_DECAY = 2e-4  # Increased weight decay for stronger regularization
    DROPOUT = 0.4  # Higher dropout to prevent overfitting
    LABEL_SMOOTHING = 0.0  # Disabled - can interfere with learning early on
    EARLY_STOPPING_PATIENCE = 25  # More patience with larger dataset
    VAL_SPLIT = 0.2  # 20% of training data for validation
    
    # Model architecture - POWERFUL but regularized (50k samples)
    # Increased capacity with strong regularization to prevent overfitting
    HIDDEN_DIM = 256  # Larger hidden dimension for more capacity
    NUM_HEADS = 8  # More attention heads for richer representations
    NUM_LAYERS = 3  # Deeper network (3 layers) for complex interactions
    
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
    data_path = Path('data/cdisc/clinical_trial_50k.csv')
    
    if not data_path.exists():
        print(f"❌ Data not found at {data_path}")
        print("   Please ensure clinical_trial_50k.csv is available.")
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
    
    # Verify data generation is working correctly
    print("\n[5.5/7] Verifying data generation...")
    try:
        # Sample a few batches to check data quality
        sample_batch = next(iter(train_loader))
        patient_state = sample_batch['patient_state']
        drug_emb = sample_batch['drug_emb']
        lab_delta = sample_batch['lab_delta']
        
        print(f"  ✓ Sample batch shapes:")
        print(f"    - Patient state: {patient_state.shape}")
        print(f"    - Drug embedding: {drug_emb.shape}")
        print(f"    - Lab delta: {lab_delta.shape}")
        
        # Check for NaN or Inf
        has_nan = torch.isnan(lab_delta).any().item()
        has_inf = torch.isinf(lab_delta).any().item()
        if has_nan or has_inf:
            print(f"  ⚠️  WARNING: Found NaN or Inf in lab_delta!")
        else:
            print(f"  ✓ No NaN or Inf values detected")
        
        # Check variance per feature
        lab_delta_np = lab_delta.numpy()
        feature_vars = np.var(lab_delta_np, axis=0)
        feature_means = np.mean(np.abs(lab_delta_np), axis=0)
        zero_var_features = [LAB_BIOMARKER_FEATURES[i] for i in range(len(feature_vars)) if feature_vars[i] < 1e-6]
        if zero_var_features:
            print(f"  ⚠️  WARNING: {len(zero_var_features)} features have near-zero variance (all zeros or constant):")
            for feat in zero_var_features[:10]:
                idx = LAB_BIOMARKER_FEATURES.index(feat)
                print(f"      - {feat}: var={feature_vars[idx]:.2e}, mean|abs|={feature_means[idx]:.4f}")
            print(f"    These features cannot be learned (R² will be 0 or undefined)")
            print(f"    Check if CSV has {[f'{f}_delta' for f in zero_var_features[:5]]} columns")
        else:
            print(f"  ✓ All features have non-zero variance")
        
        # Check value ranges (should be reasonable)
        lab_delta_min = lab_delta.min().item()
        lab_delta_max = lab_delta.max().item()
        lab_delta_mean = lab_delta.mean().item()
        lab_delta_std = lab_delta.std().item()
        print(f"  ✓ Lab delta stats: mean={lab_delta_mean:.3f}, std={lab_delta_std:.3f}, "
              f"min={lab_delta_min:.3f}, max={lab_delta_max:.3f}")
        
        if abs(lab_delta_mean) > 100 or lab_delta_std > 100:
            print(f"  ⚠️  WARNING: Lab delta values seem unusually large!")
        
    except Exception as e:
        print(f"  ⚠️  Could not verify data: {e}")
        import traceback
        traceback.print_exc()
    
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
    
    # 7. Fit normalization scalers on training data (CRITICAL for learning)
    print("\n[7/10] Fitting normalization scalers on training data...")
    # Collect training data for scaler fitting
    train_patient_data = []
    train_lab_data = []
    train_drug_embeddings = []
    
    # Sample a subset for scaler fitting (faster, representative)
    sample_size = min(5000, len(train_subset))
    sample_indices = np.random.choice(len(train_subset), size=sample_size, replace=False)
    sample_subset = torch.utils.data.Subset(train_dataset, sample_indices)
    sample_loader = DataLoader(sample_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    for batch in sample_loader:
        train_patient_data.append(batch['patient_state'].cpu().numpy())
        train_lab_data.append(batch['lab_delta'].cpu().numpy())
        train_drug_embeddings.append(batch['drug_emb'].cpu().numpy())
    
    train_patient_data = np.vstack(train_patient_data)
    train_lab_data = np.vstack(train_lab_data)
    train_drug_embeddings = np.vstack(train_drug_embeddings)
    
    print(f"  Collected {len(train_patient_data):,} samples for scaler fitting")
    
    # 8. Initialize model with POWERFUL ARCHITECTURE + STRONG REGULARIZATION
    print("\n[8/10] Initializing PharmacodynamicPredictor (powerful architecture with regularization)...")
    print(f"  Architecture: hidden_dim={HIDDEN_DIM}, num_heads={NUM_HEADS}, num_layers={NUM_LAYERS}, dropout={DROPOUT}")
    print(f"  Regularization: weight_decay={WEIGHT_DECAY}, label_smoothing={LABEL_SMOOTHING}, grad_clip=0.5")
    predictor = PharmacodynamicPredictor(
        predictor_type='cross_attention',
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_constraints=False,  # Disable during training
        device=DEVICE
    )
    
    # Calculate and print model size
    total_params = sum(p.numel() for p in predictor.model.parameters())
    trainable_params = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
    print(f"  ✓ Model initialized: {total_params / 1e6:.2f}M total parameters ({trainable_params / 1e6:.2f}M trainable)")
    print(f"  ✓ Powerful architecture with strong regularization to prevent overfitting")
    
    # Fit scalers on training data (CRITICAL - enables normalization)
    predictor.fit_scalers(
        train_patient_data=train_patient_data,
        train_lab_data=train_lab_data,
        train_drug_embeddings=train_drug_embeddings
    )
    
    # 9. Initialize constraint loss (REDUCED weights to allow better learning)
    print("\n[9/10] Initializing pharmacology constraint loss (reduced weights for better learning)...")
    criterion = PharmacologyConstraintLoss(
        lambda_monotonicity=0.005,  # Reduced - let model learn first, then enforce
        lambda_mechanism=0.02,      # Reduced - mechanism constraints can interfere early
        lambda_consistency=0.01,    # Reduced - consistency less important initially
        lambda_bounds=0.01,         # Reduced - bounds can be enforced later
        drug_encoder=drug_encoder  # Pass encoder to compute real mechanism signatures
    ).to(DEVICE)
    
    # Base MSE loss with label smoothing
    base_criterion = nn.MSELoss()
    
    # Optimizer with strong regularization
    optimizer = optim.AdamW(
        predictor.model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,  # Strong weight decay
        betas=(0.9, 0.999),  # Standard AdamW betas
        eps=1e-8
    )
    
    # Learning rate scheduler with warmup and cosine annealing
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, OneCycleLR
    
    # Warmup + cosine annealing with restarts for better convergence
    def lr_lambda(epoch):
        warmup_epochs = 10  # Longer warmup for larger model
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
    best_val_smape = float('inf')
    best_test_smape = float('inf')
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
            patient_state_raw = batch['patient_state'].to(DEVICE)
            drug_emb_raw = batch['drug_emb'].to(DEVICE)
            lab_delta_true_raw = batch['lab_delta'].to(DEVICE)
            
            # Normalize inputs for training
            if predictor.feature_scaler is not None:
                patient_state_np = patient_state_raw.cpu().numpy()
                patient_state_np = predictor.feature_scaler.transform(patient_state_np)
                patient_state = torch.tensor(patient_state_np, dtype=torch.float32, device=DEVICE)
            else:
                patient_state = patient_state_raw
            
            if predictor.drug_scaler is not None:
                drug_emb_np = drug_emb_raw.cpu().numpy()
                drug_emb_np = predictor.drug_scaler.transform(drug_emb_np)
                drug_emb = torch.tensor(drug_emb_np, dtype=torch.float32, device=DEVICE)
            else:
                drug_emb = drug_emb_raw
            
            if predictor.lab_scaler is not None:
                lab_delta_true_np = lab_delta_true_raw.cpu().numpy()
                lab_delta_true_np = predictor.lab_scaler.transform(lab_delta_true_np)
                lab_delta_true = torch.tensor(lab_delta_true_np, dtype=torch.float32, device=DEVICE)
            else:
                lab_delta_true = lab_delta_true_raw
            
            optimizer.zero_grad()
            lab_delta_pred_normalized, _ = predictor.model(patient_state, drug_emb, return_attention=False)
            
            # Base loss on normalized values (much better for training)
            base_loss = base_criterion(lab_delta_pred_normalized, lab_delta_true)
            
            # Apply label smoothing if enabled (regularization)
            if LABEL_SMOOTHING > 0:
                # Smooth targets: mix true labels with small noise
                noise_scale = lab_delta_true.std() * 0.1  # Small noise
                noise = torch.randn_like(lab_delta_true) * noise_scale
                smoothed_target = (1 - LABEL_SMOOTHING) * lab_delta_true + LABEL_SMOOTHING * noise
                base_loss = base_criterion(lab_delta_pred_normalized, smoothed_target)
            
            # Denormalize predictions for constraint loss (constraints work on original scale)
            if predictor.lab_scaler is not None:
                lab_delta_pred_np = lab_delta_pred_normalized.detach().cpu().numpy()
                lab_delta_pred_denorm = predictor.lab_scaler.inverse_transform(lab_delta_pred_np)
                lab_delta_pred_denorm = torch.tensor(lab_delta_pred_denorm, dtype=torch.float32, device=DEVICE)
            else:
                lab_delta_pred_denorm = lab_delta_pred_normalized.detach()
            
            # Constraint loss on denormalized values
            constraint_loss, components = criterion(
                lab_delta_pred_denorm, lab_delta_true_raw,
                drug_emb_raw, patient_state_raw,  # Use raw for constraints
                return_components=True
            )
            
            # Combined loss: Use normalized base_loss as primary
            # Constraint loss is secondary (already has reduced weights)
            # The constraint_loss includes its own base MSE, but we use our normalized one
            # So we use only the constraint components (excluding the base MSE from constraint_loss)
            constraint_base = components.get('base', 0.0)
            if isinstance(constraint_loss, torch.Tensor):
                constraint_only = constraint_loss - torch.tensor(constraint_base, device=constraint_loss.device, dtype=constraint_loss.dtype)
            else:
                constraint_only = constraint_loss - constraint_base
            
            loss = base_loss + constraint_only
            
            loss.backward()
            # Moderate gradient clipping for stability (allows more learning)
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), 1.0)  # Less restrictive
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
        val_preds_denorm = []  # For sMAPE calculation on original scale
        val_targets_denorm = []  # For sMAPE calculation on original scale
        
        with torch.no_grad():
            for batch in val_loader:
                patient_state_raw = batch['patient_state'].to(DEVICE)
                drug_emb_raw = batch['drug_emb'].to(DEVICE)
                lab_delta_true_raw = batch['lab_delta'].to(DEVICE)
                
                # Normalize inputs
                if predictor.feature_scaler is not None:
                    patient_state_np = patient_state_raw.cpu().numpy()
                    patient_state_np = predictor.feature_scaler.transform(patient_state_np)
                    patient_state = torch.tensor(patient_state_np, dtype=torch.float32, device=DEVICE)
                else:
                    patient_state = patient_state_raw
                
                if predictor.drug_scaler is not None:
                    drug_emb_np = drug_emb_raw.cpu().numpy()
                    drug_emb_np = predictor.drug_scaler.transform(drug_emb_np)
                    drug_emb = torch.tensor(drug_emb_np, dtype=torch.float32, device=DEVICE)
                else:
                    drug_emb = drug_emb_raw
                
                if predictor.lab_scaler is not None:
                    lab_delta_true_np = lab_delta_true_raw.cpu().numpy()
                    lab_delta_true_np = predictor.lab_scaler.transform(lab_delta_true_np)
                    lab_delta_true = torch.tensor(lab_delta_true_np, dtype=torch.float32, device=DEVICE)
                else:
                    lab_delta_true = lab_delta_true_raw
                
                lab_delta_pred_normalized, _ = predictor.model(patient_state, drug_emb, return_attention=False)
                loss = nn.MSELoss()(lab_delta_pred_normalized, lab_delta_true)
                val_loss += loss.item()
                
                # Store normalized for R² calculation
                val_preds.append(lab_delta_pred_normalized.cpu().numpy())
                val_targets.append(lab_delta_true.cpu().numpy())
                
                # Denormalize for sMAPE calculation
                if predictor.lab_scaler is not None:
                    pred_denorm = predictor.lab_scaler.inverse_transform(lab_delta_pred_normalized.cpu().numpy())
                    val_preds_denorm.append(pred_denorm)
                    val_targets_denorm.append(lab_delta_true_raw.cpu().numpy())
                else:
                    val_preds_denorm.append(lab_delta_pred_normalized.cpu().numpy())
                    val_targets_denorm.append(lab_delta_true_raw.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_preds_denorm = np.vstack(val_preds_denorm)
        val_targets_denorm = np.vstack(val_targets_denorm)
        val_r2 = r2_score(val_targets, val_preds)
        
        val_mae = mean_absolute_error(val_targets, val_preds)
        
        # Calculate symmetric MAPE (sMAPE) on denormalized values - more accurate for near-zero deltas
        # sMAPE = mean(|pred - true| / (|pred| + |true| + epsilon)) * 100
        # This handles near-zero values much better than regular MAPE
        val_errors = np.abs(val_preds_denorm - val_targets_denorm)
        val_denom = np.abs(val_preds_denorm) + np.abs(val_targets_denorm) + 1e-6
        val_smape = float(np.mean(val_errors / val_denom) * 100)
        
        # Evaluate on NOVEL drugs (test set)
        test_loss = 0.0
        test_preds_normalized = []
        test_targets_normalized = []
        test_preds_denorm = []  # For sMAPE calculation on original scale
        test_targets_denorm = []  # For sMAPE calculation on original scale
        
        with torch.no_grad():
            for batch in test_loader:
                patient_state_raw = batch['patient_state'].to(DEVICE)
                drug_emb_raw = batch['drug_emb'].to(DEVICE)
                lab_delta_true_raw = batch['lab_delta'].to(DEVICE)
                
                # Normalize inputs
                if predictor.feature_scaler is not None:
                    patient_state_np = patient_state_raw.cpu().numpy()
                    patient_state_np = predictor.feature_scaler.transform(patient_state_np)
                    patient_state = torch.tensor(patient_state_np, dtype=torch.float32, device=DEVICE)
                else:
                    patient_state = patient_state_raw
                
                if predictor.drug_scaler is not None:
                    drug_emb_np = drug_emb_raw.cpu().numpy()
                    drug_emb_np = predictor.drug_scaler.transform(drug_emb_np)
                    drug_emb = torch.tensor(drug_emb_np, dtype=torch.float32, device=DEVICE)
                else:
                    drug_emb = drug_emb_raw
                
                if predictor.lab_scaler is not None:
                    lab_delta_true_np = lab_delta_true_raw.cpu().numpy()
                    lab_delta_true_np = predictor.lab_scaler.transform(lab_delta_true_np)
                    lab_delta_true = torch.tensor(lab_delta_true_np, dtype=torch.float32, device=DEVICE)
                else:
                    lab_delta_true = lab_delta_true_raw
                
                lab_delta_pred_normalized, _ = predictor.model(patient_state, drug_emb, return_attention=False)
                loss = nn.MSELoss()(lab_delta_pred_normalized, lab_delta_true)
                test_loss += loss.item()
                
                # Store normalized for R² calculation
                test_preds_normalized.append(lab_delta_pred_normalized.cpu().numpy())
                test_targets_normalized.append(lab_delta_true.cpu().numpy())
                
                # Denormalize for sMAPE calculation
                if predictor.lab_scaler is not None:
                    pred_denorm = predictor.lab_scaler.inverse_transform(lab_delta_pred_normalized.cpu().numpy())
                    test_preds_denorm.append(pred_denorm)
                    test_targets_denorm.append(lab_delta_true_raw.cpu().numpy())
                else:
                    test_preds_denorm.append(lab_delta_pred_normalized.cpu().numpy())
                    test_targets_denorm.append(lab_delta_true_raw.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_preds_normalized = np.vstack(test_preds_normalized)
        test_targets_normalized = np.vstack(test_targets_normalized)
        test_preds_denorm = np.vstack(test_preds_denorm)
        test_targets_denorm = np.vstack(test_targets_denorm)
        
        # Calculate metrics on normalized data (better for comparing across scales)
        test_r2 = r2_score(test_targets_normalized, test_preds_normalized)
        test_mae = mean_absolute_error(test_targets_normalized, test_preds_normalized)
        
        # Calculate symmetric MAPE (sMAPE) on denormalized values - more accurate for near-zero deltas
        # sMAPE = mean(|pred - true| / (|pred| + |true| + epsilon)) * 100
        # This handles near-zero values much better than regular MAPE
        test_errors = np.abs(test_preds_denorm - test_targets_denorm)
        test_denom = np.abs(test_preds_denorm) + np.abs(test_targets_denorm) + 1e-6
        test_smape = float(np.mean(test_errors / test_denom) * 100)
        
        # Update scheduler (per epoch)
        scheduler.step()
        
        # Also compute per-feature R² to see which biomarkers are learning
        if epoch % 20 == 0:
            per_feature_r2 = []
            per_feature_stats = []
            # Use test set for per-feature analysis (novel drugs)
            for i in range(test_targets_normalized.shape[1]):
                try:
                    feat_targets = test_targets_normalized[:, i]
                    feat_preds = test_preds_normalized[:, i]
                    
                    # Check for low variance (can cause R² issues)
                    target_var = np.var(feat_targets)
                    target_mean = np.mean(np.abs(feat_targets))
                    target_std = np.std(feat_targets)
                    
                    # Skip R² calculation if variance is too low (will be undefined/0)
                    if target_var < 1e-8:
                        per_feature_r2.append((LAB_BIOMARKER_FEATURES[i], 0.0))
                        per_feature_stats.append({
                            'name': LAB_BIOMARKER_FEATURES[i],
                            'r2': 0.0,
                            'mae': 0.0,
                            'target_var': target_var,
                            'target_mean_abs': target_mean,
                            'target_std': target_std,
                            'note': 'ZERO_VARIANCE'
                        })
                        continue
                    
                    # Calculate R²
                    feat_r2 = r2_score(feat_targets, feat_preds)
                    
                    # Calculate MAE for context
                    feat_mae = mean_absolute_error(feat_targets, feat_preds)
                    
                    per_feature_r2.append((LAB_BIOMARKER_FEATURES[i], feat_r2))
                    per_feature_stats.append({
                        'name': LAB_BIOMARKER_FEATURES[i],
                        'r2': feat_r2,
                        'mae': feat_mae,
                        'target_var': target_var,
                        'target_mean_abs': target_mean,
                        'target_std': target_std
                    })
                except Exception as e:
                    per_feature_r2.append((LAB_BIOMARKER_FEATURES[i], -999))
                    per_feature_stats.append({
                        'name': LAB_BIOMARKER_FEATURES[i],
                        'error': str(e)
                    })
            
            # Show top 3 and bottom 3 features with diagnostics
            per_feature_r2.sort(key=lambda x: x[1], reverse=True)
            print(f"    Top 3 features: {per_feature_r2[:3]}")
            print(f"    Bottom 3 features: {per_feature_r2[-3:]}")
            
            # Show diagnostics for problematic features
            if epoch == 20:  # Only print detailed diagnostics on first check
                print(f"\n    Data Generation Diagnostics (first check):")
                
                # Check for zero-variance features
                zero_var = [s for s in per_feature_stats if s.get('note') == 'ZERO_VARIANCE']
                if zero_var:
                    print(f"    ⚠️  {len(zero_var)} features have ZERO VARIANCE (all values are the same):")
                    for stat in zero_var[:10]:
                        print(f"      - {stat['name']}: All values = {stat.get('target_mean_abs', 0):.4f}")
                    print(f"      These features cannot be learned! Check CSV for missing *_delta columns.")
                    expected_cols = [f"{s['name']}_delta" for s in zero_var[:5]]
                    print(f"      Expected columns: {expected_cols}")
                
                # Check for problematic features with negative R²
                problematic = [s for s in per_feature_stats if s.get('r2', 0) < -1.0 and s.get('note') != 'ZERO_VARIANCE']
                if problematic:
                    print(f"\n    ⚠️  {len(problematic)} features with R² < -1.0 (model worse than mean):")
                    for stat in problematic[:5]:  # Show first 5
                        print(f"      {stat['name']}: R²={stat.get('r2', 'N/A'):.2f}, "
                              f"MAE={stat.get('mae', 'N/A'):.4f}, "
                              f"Target Std={stat.get('target_std', 'N/A'):.4f}")
                    print(f"      This suggests the model is struggling to learn these features.")
                
                # Check for features with very low variance (might indicate data gen issue)
                low_var = [s for s in per_feature_stats if 1e-8 <= s.get('target_var', 1e10) < 1e-4 and s.get('note') != 'ZERO_VARIANCE']
                if low_var:
                    print(f"\n    ⚠️  {len(low_var)} features have very low variance (<1e-4):")
                    for stat in low_var[:5]:
                        print(f"      {stat['name']}: var={stat.get('target_var', 0):.2e}, std={stat.get('target_std', 0):.4f}")
        
        # Early stopping and model saving based on validation loss (lower is better)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_r2 = val_r2
            best_test_r2 = test_r2
            best_val_smape = val_smape
            best_test_smape = test_smape
            best_test_loss = test_loss
            best_epoch = epoch
            patience_counter = 0
            predictor.save(
                'models/pharmacodynamic_predictor/predictor_novel_drug_best.pt',
                trained=True,
                training_metrics={
                    'best_epoch': best_epoch,
                    'val_loss': float(val_loss),
                    'val_r2': float(val_r2),
                    'val_smape_pct': float(val_smape),
                    'test_loss': float(test_loss),
                    'test_r2': float(test_r2),
                    'test_smape_pct': float(test_smape),
                }
            )
            print(f"  ✓ New best model! Val Loss: {val_loss:.4f} (saved)")
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else LEARNING_RATE
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} (sMAPE: {val_smape:.1f}%) | "
                  f"Test: {test_loss:.4f} (sMAPE: {test_smape:.1f}%) | "
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
            print(f"   Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
            break
    
    total_time = time.time() - start_time
    
    print(f"\n✓ Training complete in {total_time/60:.1f} minutes!")
    print(f"✓ Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"✓ Best test loss: {best_test_loss:.4f} (epoch {best_epoch})")
    print(f"  (sMAPE: Val={best_val_smape:.1f}%, Test={best_test_smape:.1f}%, R²: Val={best_val_r2:.4f}, Test={best_test_r2:.4f})")
    print(f"\nTarget: sMAPE < 20% (symmetric mean absolute percentage error < 20%)")
    
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
