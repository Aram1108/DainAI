"""
SIMPLE CONDITIONAL GAN FOR SYNTHETIC PATIENT GENERATION

This module implements a straightforward Conditional GAN that generates
realistic synthetic patients based on user inputs (age, sex, height, weight).

The GAN learns to create complete medical profiles including:
- 8 body measurements (BMI, arm circumference, etc.)
- 22 lab results (blood work)
- 7 questionnaire responses
- Total: 37 generated features

Think of it like: User provides basic info → Model creates complete patient data
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.lab_reference_ranges import (
    LAB_RANGES, CRITICAL_RULES, clamp_to_physiological_limits
)



# ============================================================
# FEATURE DEFINITIONS
# ============================================================

# User provides these 4 inputs
USER_INPUT_COLS = ['AGE', 'SEX', 'BMXHT', 'BMXWT']

# Model generates these 44 features (8 + 22 + 7 + 7)
BODY_MEASUREMENT_COLS = [
    'BMXARMC', 'BMXARML', 'BMXBMI', 'BMXLEG', 
    'BMXSUB', 'BMXTHICR', 'BMXTRI', 'BMXWAIST'
]

LAB_RESULT_COLS = [
    'LBDSCASI', 'LBDSCH', 'LBDSCR', 'LBDTC',
    'LBXBCD', 'LBXBPB', 'LBXCRP', 'LBXSAL', 'LBXSAS',
    'LBXSBU', 'LBXSCA', 'LBXSCH', 'LBXSCL', 'LBXSGB',
    'LBXSGL', 'LBXSGT', 'LBXSK', 'LBXSNA', 'LBXSOS',
    'LBXSTP', 'LBXSUA', 'LBXTC'
]

QUESTIONNAIRE_COLS = [
    'MCQ160B', 'MCQ160C', 'MCQ160E', 'MCQ160F',
    'MCQ160K', 'MCQ160L', 'MCQ220'
]

# All generated features (total: 37 = 8 body + 22 lab + 7 questionnaire)
GENERATED_FEATURE_COLS = BODY_MEASUREMENT_COLS + LAB_RESULT_COLS + QUESTIONNAIRE_COLS


# ============================================================
# GENERATOR NETWORK
# ============================================================

class Generator(nn.Module):
    """
    GENERATOR: Creates synthetic patients from conditions + noise
    
    Input: 
        - 4 conditions (age, sex, height, weight) - what user provides
        - 100 noise dimensions - for variety between patients
    
    Output:
        - 37 features (8 body + 22 lab + 7 questionnaire)
    
    Architecture:
        104 → 512 → 512 → 256 → 37
        
    Think of it like: Given basic demographics + randomness → complete medical profile
    """
    
    def __init__(self, noise_dim=100):
        super().__init__()
        
        self.noise_dim = noise_dim
        
        # Input: 4 conditions + 100 noise = 104
        # Output: 37 generated features (8 body + 22 lab + 7 questionnaire)
        
        self.network = nn.Sequential(
            # Layer 1: Expand from 104 to 512 (MATCH discriminator power)
            nn.Linear(104, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Layer 2: Stay at 512
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Layer 3: Taper down to 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Output layer: 256 → 37 features
            nn.Linear(256, 37),
            nn.Tanh()  # Output range: -1 to 1 (denormalized later)
        )
    
    def forward(self, conditions, noise):
        """
        Generate synthetic patient features
        
        Args:
            conditions: [batch, 4] - normalized age/sex/height/weight
            noise: [batch, 100] - random noise for patient variety
        
        Returns:
            features: [batch, 44] - generated patient features
        """
        # Combine conditions with noise
        x = torch.cat([conditions, noise], dim=1)
        
        # Generate features
        features = self.network(x)
        
        return features


# ============================================================
# DISCRIMINATOR NETWORK
# ============================================================

class Discriminator(nn.Module):
    """
    DISCRIMINATOR: Judges if patient data is real or fake
    
    Input:
        - 4 conditions (age, sex, height, weight)
        - 37 features (body + lab + questionnaire)
        Total: 41 dimensions
    
    Output:
        - 1 score (probability that patient is real)
    
    Architecture:
        41 → 256 → 128 → 1
        
    Think of it like: A medical expert checking if patient data makes sense
    """
    
    def __init__(self):
        super().__init__()
        
        # Input: 4 conditions + 37 features = 41
        # Output: 1 (real/fake probability)
        
        self.network = nn.Sequential(
            # Layer 1: 41 → 256 (REDUCED from 512 to weaken discriminator)
            nn.Linear(41, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),  # INCREASED from 0.3 to slow down discriminator
            
            # Layer 2: 256 → 128
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),  # INCREASED from 0.3
            
            # Output layer: 128 → 1
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability: 0 to 1
        )
    
    def forward(self, conditions, features):
        """
        Judge if patient is real or fake
        
        Args:
            conditions: [batch, 4] - age/sex/height/weight
            features: [batch, 37] - patient features (real or generated)
        
        Returns:
            probability: [batch, 1] - how likely patient is real (0-1)
        """
        # Combine conditions with features
        x = torch.cat([conditions, features], dim=1)
        
        # Judge realness
        probability = self.network(x)
        
        return probability


# ============================================================
# MAIN PATIENT GENERATOR CLASS
# ============================================================

class PatientGenerator:
    """
    MAIN CLASS: Simple interface for generating synthetic patients
    
    This is what users interact with. Makes everything easy!
    
    Usage:
        # Step 1: Load and train
        gen = PatientGenerator(data_path='preprocessed_nhanes/nhanes_final_complete.csv')
        gen.train(epochs=200)
        
        # Step 2: Generate patients
        patients = gen.generate(age=45, sex='M', height=175, weight=80, n=10)
    """
    
    def __init__(self, data_path='preprocessed_nhanes/nhanes_final_complete.csv', device=None):
        """
        Initialize the patient generator
        
        Args:
            data_path: Path to the preprocessed CSV file
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        # Setup device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"[INIT] Using device: {self.device}")
        
        # Load data
        print(f"[INIT] Loading data from {data_path}...")
        try:
            self.df = pd.read_csv(data_path)
            print(f"[OK] Loaded {len(self.df):,} patients")
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")
        
        # Verify required columns exist
        required_cols = USER_INPUT_COLS + GENERATED_FEATURE_COLS
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Store column names
        self.condition_cols = USER_INPUT_COLS
        self.feature_cols = GENERATED_FEATURE_COLS
        
        print(f"[OK] Feature breakdown:")
        print(f"  - User inputs (conditions): {len(self.condition_cols)}")
        print(f"  - Body measurements: {len(BODY_MEASUREMENT_COLS)}")
        print(f"  - Lab results: {len(LAB_RESULT_COLS)}")
        print(f"  - Questionnaire: {len(QUESTIONNAIRE_COLS)}")
        print(f"  - Total to generate: {len(self.feature_cols)}")
        
        # Helper: force a column to float64 (handles StringDtype, object, int, etc.)
        def _to_float64(col_name, string_map=None):
            col = self.df[col_name]
            if pd.api.types.is_numeric_dtype(col):
                self.df[col_name] = col.astype(np.float64)
            else:
                if string_map:
                    self.df[col_name] = col.astype(str).str.strip().map(string_map)
                else:
                    self.df[col_name] = pd.to_numeric(col.astype(str).str.strip(), errors='coerce')
                med = self.df[col_name].median() if self.df[col_name].notna().any() else 0
                self.df[col_name] = self.df[col_name].fillna(med).astype(np.float64)
        
        # Encode SEX to numeric (M=0, F=1)
        if 'SEX' in self.df.columns:
            _to_float64('SEX', string_map={'M': 0.0, 'F': 1.0, 'Male': 0.0, 'Female': 1.0})
        
        # Force all condition columns to numeric
        for c in self.condition_cols:
            if c in self.df.columns:
                _to_float64(c)
        
        # Calculate normalization statistics
        self.condition_mean = self.df[self.condition_cols].mean()
        self.condition_std = self.df[self.condition_cols].std().replace(0, 1.0)
        
        # Force all feature columns to numeric
        for c in self.feature_cols:
            if c in self.df.columns:
                _to_float64(c)
        
        self.feature_mean = self.df[self.feature_cols].mean()
        self.feature_std = self.df[self.feature_cols].std().replace(0, 1.0)
        
        print(f"[OK] Normalization stats calculated")
        print(f"  - Age: {self.condition_mean['AGE']:.1f} ± {self.condition_std['AGE']:.1f}")
        print(f"  - Height: {self.condition_mean['BMXHT']:.1f} ± {self.condition_std['BMXHT']:.1f} cm")
        print(f"  - Weight: {self.condition_mean['BMXWT']:.1f} ± {self.condition_std['BMXWT']:.1f} kg")
        
        # Initialize GAN models
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # Check if pretrained model exists (try best model first, then fallback)
        model_dir = Path('models/generator')
        model_paths = [
            model_dir / 'patient_generator_best.pt',           # Generator-only best model
            model_dir / 'patient_generator_gan_best.pt',      # Full GAN best model
            model_dir / 'patient_generator_gan.pt',           # Legacy full model
        ]
        
        self.model_path = None
        loaded = False
        
        for model_path in model_paths:
            if model_path.exists():
                try:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    self.generator.load_state_dict(checkpoint['generator'])
                    if 'discriminator' in checkpoint:
                        self.discriminator.load_state_dict(checkpoint['discriminator'])
                    
                    self.model_path = model_path
                    loaded = True
                    
                    epoch = checkpoint.get('epoch', 'unknown')
                    fid = checkpoint.get('fid_score', None)
                    
                    print(f"[OK] Loaded pretrained model from {model_path}")
                    print(f"  Training epochs: {epoch}")
                    if fid is not None:
                        print(f"  FID score: {fid:.2f} (lower is better)")
                    break
                except Exception as e:
                    print(f"[WARNING] Failed to load {model_path}: {e}")
                    continue
        
        if not loaded:
            self.model_path = model_dir / 'patient_generator_gan.pt'  # Default path for saving
            print(f"[INFO] No pretrained model found in:")
            for path in model_paths:
                print(f"  - {path}")
            print("  Run train() method to train the model")
        
        print("="*70)
    
    def _compute_fid(self, real_features, fake_features):
        """
        Compute Fréchet Inception Distance (FID) between real and generated distributions
        
        FID is the gold standard for evaluating GAN quality:
        - FID = 0: Distributions are identical (perfect generation)
        - Lower FID: Generated distribution closer to real distribution
        - Higher FID: More difference between distributions
        
        Why FID is better than loss:
        - Measures both quality AND diversity
        - Detects mode collapse (if generator produces limited variety)
        - Correlates strongly with human perception
        - Industry standard for GAN evaluation
        
        FID = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2√(Σ_real·Σ_fake))
        
        Args:
            real_features: Real patient features [N, D]
            fake_features: Generated patient features [N, D]
        
        Returns:
            float: FID score (lower is better, <50 is good, <10 is excellent)
        """
        # Convert to numpy for numerical stability
        real = real_features.cpu().numpy()
        fake = fake_features.cpu().numpy()
        
        # Compute mean and covariance
        mu_real = np.mean(real, axis=0)
        mu_fake = np.mean(fake, axis=0)
        
        sigma_real = np.cov(real, rowvar=False)
        sigma_fake = np.cov(fake, rowvar=False)
        
        # Compute squared difference of means
        diff = mu_real - mu_fake
        mean_diff = np.sum(diff ** 2)
        
        # Compute sqrt of product of covariances using scipy
        from scipy import linalg
        
        # Add small epsilon to diagonal for numerical stability
        eps = 1e-6
        sigma_real += np.eye(sigma_real.shape[0]) * eps
        sigma_fake += np.eye(sigma_fake.shape[0]) * eps
        
        # Compute matrix square root
        covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
        
        # Check for imaginary components (numerical issues)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # Compute trace
        trace_term = np.trace(sigma_real + sigma_fake - 2 * covmean)
        
        # FID formula
        fid = mean_diff + trace_term
        
        return float(fid)
    
    def train(self, epochs=200, batch_size=128, lr=0.0002, 
              save_every=50, show_progress_every=10, early_stopping_patience=30):
        """
        Train the GAN to generate realistic patients
        
        How it works:
        1. Generator creates fake patients
        2. Discriminator tries to tell real from fake
        3. Generator gets better at fooling discriminator
        4. Discriminator gets better at spotting fakes
        5. Eventually, generator creates very realistic patients!
        
        Args:
            epochs: Number of training iterations (200 is usually good)
            batch_size: How many patients to process at once
            lr: Learning rate (0.0002 is good for GANs)
            save_every: Save checkpoint every N epochs
            show_progress_every: Print progress every N epochs
            early_stopping_patience: Stop training if no FID improvement for N epochs (None to disable)
        """
        print("\n" + "="*70)
        print("TRAINING PATIENT GENERATOR (Conditional GAN)")
        print("="*70)
        print(f"Dataset: {len(self.df):,} real patients from NHANES")
        print(f"Training for {epochs} epochs with batch size {batch_size}")
        print(f"Total training batches per epoch: {len(self.df) // batch_size:,}")
        print(f"Samples per epoch: {(len(self.df) // batch_size) * batch_size:,}")
        print("This may take a while...")
        print("="*70 + "\n")
        
        # Setup optimizers (separate for generator and discriminator)
        # Generator gets HIGHER learning rate to compensate for architecture disadvantage
        optimizer_g = torch.optim.Adam(
            self.generator.parameters(), 
            lr=lr * 1.5,  # 1.5x higher for generator (0.0003 vs 0.0002)
            betas=(0.5, 0.999)
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=lr,  # Standard rate for discriminator
            betas=(0.5, 0.999)
        )
        
        # Learning rate schedulers to prevent loss jumps
        scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_g, mode='min', factor=0.5, patience=20, verbose=True
        )
        scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_d, mode='min', factor=0.5, patience=20, verbose=True
        )
        
        # Track best model using evaluation metrics (not just losses)
        best_fid_score = float('inf')  # Lower is better - FID measures distribution similarity
        best_epoch = 0
        epochs_since_best = 0  # Track epochs without improvement for early stopping
        eval_interval = max(10, show_progress_every)  # Evaluate every N epochs
        
        # Exponential moving average for smoother loss tracking
        ema_g_loss = None
        ema_d_loss = None
        ema_alpha = 0.1  # Smoothing factor
        
        # Loss function: Binary Cross Entropy
        criterion = nn.BCELoss()
        
        # Prepare training data
        conditions = self.df[self.condition_cols].values
        features = self.df[self.feature_cols].values
        
        # Normalize data (helps training)
        conditions = (conditions - self.condition_mean.values) / self.condition_std.values
        features = (features - self.feature_mean.values) / self.feature_std.values
        
        # Convert to PyTorch tensors
        conditions = torch.tensor(conditions, dtype=torch.float32)
        features = torch.tensor(features, dtype=torch.float32)
        
        # Calculate batches per epoch
        n_samples = len(conditions)
        batches_per_epoch = n_samples // batch_size
        
        print(f"[DATASET] Loaded {n_samples:,} patients")
        print(f"[TRAINING] {batches_per_epoch:,} batches per epoch")
        print(f"[TRAINING] {batches_per_epoch * batch_size:,} samples per epoch")
        print(f"[TRAINING] Total updates: {epochs * batches_per_epoch:,} batches\n")
        
        # Training loop
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            # Process all batches in this epoch
            for batch_idx in range(batches_per_epoch):
                # Get random batch of real patients
                indices = torch.randint(0, len(conditions), (batch_size,))
                real_conditions = conditions[indices].to(self.device)
                real_features = features[indices].to(self.device)
            
                # Add random noise to conditions to prevent overfitting
                # This makes the model generalize better to new age/sex/height/weight combinations
                condition_noise = torch.randn_like(real_conditions) * 0.05  # 5% noise
                real_conditions_noisy = real_conditions + condition_noise
                
                # ============================================================
                # STEP 1: TRAIN DISCRIMINATOR
                # Goal: Learn to distinguish real patients from fake ones
                # ============================================================
                
                # Generate fake patients with noisy conditions (needed for both D and G training)
                noise = torch.randn(batch_size, 100, device=self.device)
                fake_features = self.generator(real_conditions_noisy, noise)
                
                # ADD NOISE TO FEATURES (prevents discriminator overfitting)
                real_features_noisy = real_features + torch.randn_like(real_features) * 0.1
                fake_features_noisy = fake_features.detach() + torch.randn_like(fake_features) * 0.1
                
                # Score real patients (should be close to 0.9 - label smoothing)
                # Use noisy features to prevent overfitting
                real_score = self.discriminator(real_conditions_noisy, real_features_noisy)
                real_labels = torch.ones_like(real_score) * 0.9  # Label smoothing
                loss_d_real = criterion(real_score, real_labels)
                
                # Score fake patients (should be close to 0.1 - label smoothing)
                fake_score = self.discriminator(real_conditions_noisy, fake_features_noisy)
                fake_labels = torch.ones_like(fake_score) * 0.1  # Label smoothing
                loss_d_fake = criterion(fake_score, fake_labels)
                
                # Total discriminator loss
                loss_d = loss_d_real + loss_d_fake
                
                # Update discriminator
                optimizer_d.zero_grad()
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                optimizer_d.step()
                
                # ============================================================
                # STEP 2: TRAIN GENERATOR (MULTIPLE TIMES)
                # Goal: Create fake patients that fool the discriminator
                # Train G 2x per D update to help it catch up
                # ============================================================
                
                for g_step in range(2):  # Train generator TWICE per discriminator update
                    # Generate new fake patients with noisy conditions
                    noise = torch.randn(batch_size, 100, device=self.device)
                    fake_features = self.generator(real_conditions_noisy, noise)
                    
                    # Try to fool discriminator (want it to output close to 1)
                    fake_score = self.discriminator(real_conditions_noisy, fake_features)
                    real_labels = torch.ones_like(fake_score)  # Want discriminator to think it's real
                    loss_g = criterion(fake_score, real_labels)
                    
                    # Update generator
                    optimizer_g.zero_grad()
                    loss_g.backward()
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                    optimizer_g.step()
                    
                    # Only accumulate loss from first G update for logging
                    if g_step == 0:
                        epoch_g_loss += loss_g.item()
                
                # Accumulate discriminator loss
                epoch_d_loss += loss_d.item()
            
            # Average losses over all batches in this epoch
            epoch_g_loss /= batches_per_epoch
            epoch_d_loss /= batches_per_epoch
            
            # ============================================================
            # EVALUATION & CHECKPOINTING
            # ============================================================
            
            # Update exponential moving average for smoother loss tracking
            if ema_g_loss is None:
                ema_g_loss = epoch_g_loss
                ema_d_loss = epoch_d_loss
            else:
                ema_g_loss = ema_alpha * epoch_g_loss + (1 - ema_alpha) * ema_g_loss
                ema_d_loss = ema_alpha * epoch_d_loss + (1 - ema_alpha) * ema_d_loss
            
            # Update learning rate schedulers
            scheduler_g.step(ema_g_loss)
            scheduler_d.step(ema_d_loss)
            
            # Compute quality metrics periodically (not every epoch - too expensive)
            if epoch % eval_interval == 0 or epoch == epochs - 1:
                # Generate evaluation samples
                self.generator.eval()
                with torch.no_grad():
                    # Use random subset of real conditions for evaluation
                    eval_indices = torch.randint(0, len(conditions), (min(1000, len(conditions)),))
                    eval_conditions = conditions[eval_indices].to(self.device)
                    eval_real_features = features[eval_indices].to(self.device)
                    
                    # Generate fake samples
                    eval_noise = torch.randn(len(eval_indices), 100, device=self.device)
                    eval_fake_features = self.generator(eval_conditions, eval_noise)
                    
                    # Compute FID (Fréchet Inception Distance) - gold standard for GAN evaluation
                    # Lower FID = better quality (generated distribution matches real)
                    # FID < 50: Good quality
                    # FID < 10: Excellent quality
                    fid_score = self._compute_fid(eval_real_features, eval_fake_features)
                    
                    # Compute feature variance - measures diversity
                    # Higher variance = more diverse generations
                    fake_variance = eval_fake_features.var(dim=0).mean().item()
                    real_variance = eval_real_features.var(dim=0).mean().item()
                    variance_ratio = fake_variance / (real_variance + 1e-8)
                    
                self.generator.train()
                
                # Print metrics
                gd_ratio = epoch_g_loss / (epoch_d_loss + 1e-8)  # Avoid division by zero
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"D_loss: {epoch_d_loss:.4f} (EMA: {ema_d_loss:.4f}) | "
                      f"G_loss: {epoch_g_loss:.4f} (EMA: {ema_g_loss:.4f}) | "
                      f"G/D ratio: {gd_ratio:.2f} | "  # Shows balance (ideal: 0.5-2.0)
                      f"FID: {fid_score:.2f} | Var: {variance_ratio:.2f} | "
                      f"Batches: {batches_per_epoch:,}")
                
                # Save best model based on FID score (not loss!)
                if fid_score < best_fid_score:
                    best_fid_score = fid_score
                    best_epoch = epoch
                    epochs_since_best = 0  # Reset counter on improvement
                    
                    # Save to models/generator/ directory
                    gen_dir = self.model_path.parent
                    gen_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save full checkpoint (generator + discriminator + optimizers)
                    best_model_path = gen_dir / 'patient_generator_gan_best.pt'
                    torch.save({
                        'epoch': epoch,
                        'generator': self.generator.state_dict(),
                        'discriminator': self.discriminator.state_dict(),
                        'optimizer_g': optimizer_g.state_dict(),
                        'optimizer_d': optimizer_d.state_dict(),
                        'fid_score': fid_score,
                        'variance_ratio': variance_ratio,
                        'g_loss': ema_g_loss,
                        'd_loss': ema_d_loss,
                    }, best_model_path)
                    
                    # Save generator-only model
                    gen_model_path = gen_dir / 'patient_generator_best.pt'
                    torch.save({
                        'epoch': epoch,
                        'generator': self.generator.state_dict(),
                        'fid_score': fid_score,
                        'variance_ratio': variance_ratio,
                    }, gen_model_path)
                    
                    # Save metadata as JSON
                    import json
                    import time
                    metadata = {
                        'model_info': {
                            'name': 'Conditional GAN Patient Generator',
                            'version': '1.0',
                            'architecture': 'Generator: 104→512→512→256→37, Discriminator: 41→256→128→1',
                            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                        },
                        'training_info': {
                            'best_epoch': epoch,
                            'total_epochs_trained': epochs,
                            'batch_size': batch_size,
                            'learning_rate': lr,
                            'optimizer': 'Adam (betas=0.5, 0.999)',
                            'noise_dim': 100,
                            'condition_noise': 0.05,
                            'early_stopping_patience': early_stopping_patience,
                        },
                        'quality_metrics': {
                            'fid_score': float(fid_score),
                            'variance_ratio': float(variance_ratio),
                            'g_loss_ema': float(ema_g_loss),
                            'd_loss_ema': float(ema_d_loss),
                        },
                        'data_info': {
                            'dataset': 'NHANES',
                            'n_patients': len(conditions),
                            'n_features_generated': len(self.feature_cols),
                            'feature_categories': {
                                'body_measurements': 8,
                                'lab_results': 22,
                                'questionnaires': 7,
                            },
                        },
                        'usage': {
                            'input_features': ['AGE', 'SEX', 'BMXHT', 'BMXWT'],
                            'output_features': self.feature_cols,
                            'normalization': {
                                'conditions_mean': self.condition_mean.to_dict(),
                                'conditions_std': self.condition_std.to_dict(),
                                'features_mean': self.feature_mean.to_dict(),
                                'features_std': self.feature_std.to_dict(),
                            },
                        },
                    }
                    
                    metadata_path = gen_dir / 'metadata.json'
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2)
                    
                    print(f"  [BEST] New best model! FID: {fid_score:.2f}")
                    print(f"  [SAVE] Generator saved to: {gen_model_path}")
                    print(f"  [SAVE] Metadata saved to: {metadata_path}")
                else:
                    # No improvement - increment counter (only count evaluation epochs)
                    if early_stopping_patience is not None:
                        epochs_since_best += eval_interval
                        
                        # Early stopping check
                        if epochs_since_best >= early_stopping_patience:
                            print(f"\n⚠ Early stopping triggered!")
                            print(f"  No FID improvement for {epochs_since_best} epochs (patience: {early_stopping_patience})")
                            print(f"  Best FID: {best_fid_score:.2f} at epoch {best_epoch}")
                            break
            
            # Print progress (without metrics - those are computed less frequently)
            elif epoch % show_progress_every == 0:
                gd_ratio = epoch_g_loss / (epoch_d_loss + 1e-8)  # Avoid division by zero
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"D_loss: {epoch_d_loss:.4f} (EMA: {ema_d_loss:.4f}) | "
                      f"G_loss: {epoch_g_loss:.4f} (EMA: {ema_g_loss:.4f}) | "
                      f"G/D ratio: {gd_ratio:.2f} | "  # Shows balance (ideal: 0.5-2.0)
                      f"Batches: {batches_per_epoch:,}")
                if early_stopping_patience is not None and epochs_since_best > 0:
                    print(f"  (No improvement for {epochs_since_best} epochs, patience: {early_stopping_patience})")
        
        # Final save
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"[OK] Best model saved to: {self.model_path.parent / 'patient_generator_gan_best.pt'}")
        print(f"[OK] Generator saved to: {self.model_path.parent / 'patient_generator_best.pt'}")
        print(f"[OK] Metadata saved to: {self.model_path.parent / 'metadata.json'}")
        print(f"  Best epoch: {best_epoch} with FID score: {best_fid_score:.2f}")
        print("  (Lower FID = better quality: <50 is good, <10 is excellent)")
        print("[OK] Ready to generate synthetic patients!")
        print("="*70 + "\n")
    
    def generate(self, age, sex, height, weight, n=1, seed=None):
        """
        Generate synthetic patients based on user inputs
        
        Args:
            age: Patient age in years (12-85)
            sex: 'M' for male or 'F' for female
            height: Height in cm (e.g., 175)
            weight: Weight in kg (e.g., 80)
            n: How many patients to generate
            seed: Random seed for reproducibility (optional)
        
        Returns:
            DataFrame with n synthetic patients (all features)
        
        Example:
            # Generate 5 middle-aged men
            patients = gen.generate(age=45, sex='M', height=175, weight=80, n=5)
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Validate inputs
        if sex not in ['M', 'F']:
            raise ValueError(f"Sex must be 'M' or 'F', got: {sex}")
        if not (12 <= age <= 85):
            print(f"[WARNING] Age {age} is outside training range (12-85)")
        if not (140 <= height <= 200):
            print(f"[WARNING] Height {height} is outside typical range (140-200 cm)")
        if not (30 <= weight <= 150):
            print(f"[WARNING] Weight {weight} is outside typical range (30-150 kg)")
        
        # Encode sex (M=0, F=1)
        sex_encoded = 0.0 if sex == 'M' else 1.0
        
        # Create condition vector
        condition = np.array([[age, sex_encoded, height, weight]])
        
        # Normalize conditions
        condition = (condition - self.condition_mean.values) / self.condition_std.values
        condition = torch.tensor(condition, dtype=torch.float32, device=self.device)
        condition = condition.repeat(n, 1)  # Repeat for n patients
        
        # Generate features
        self.generator.eval()
        with torch.no_grad():
            # Create random noise for variety
            noise = torch.randn(n, 100, device=self.device)
            
            # Generate features
            generated_features = self.generator(condition, noise)
        
        # Convert to numpy and denormalize
        generated_features = generated_features.cpu().numpy()
        generated_features = (generated_features * self.feature_std.values 
                            + self.feature_mean.values)
        
        # Apply physiological constraints: Ensure positive values for metrics that must be positive
        # This prevents impossible negative values for enzymes, proteins, etc.
        for i, col_name in enumerate(self.feature_cols):
            if col_name in CRITICAL_RULES["MUST_BE_POSITIVE"]:
                # Use ReLU-like behavior: clamp negative values to small positive value
                generated_features[:, i] = np.maximum(generated_features[:, i], 0.01)
            
            # Clamp all lab values to physiological limits
            if col_name in LAB_RANGES and LAB_RANGES[col_name].get('physiological_limit'):
                min_phys, max_phys = LAB_RANGES[col_name]['physiological_limit']
                generated_features[:, i] = np.clip(generated_features[:, i], min_phys, max_phys)
        
        # Round questionnaire responses to nearest valid value (1, 2, or 9)
        questionnaire_start_idx = len(BODY_MEASUREMENT_COLS) + len(LAB_RESULT_COLS)
        for i in range(len(QUESTIONNAIRE_COLS)):
            col_idx = questionnaire_start_idx + i
            # Round to nearest integer first
            rounded = np.round(generated_features[:, col_idx])
            # Map to valid values (1, 2, or 9)
            generated_features[:, col_idx] = np.where(
                rounded < 1.5, 1.0,
                np.where(rounded < 5.5, 2.0, 9.0)
            )
        
        # Apply BMI constraint: BMI = weight / (height/100)^2
        bmi_idx = self.feature_cols.index('BMXBMI')
        generated_features[:, bmi_idx] = weight / (height / 100) ** 2
        
        # Create DataFrame
        result = pd.DataFrame(generated_features, columns=self.feature_cols)
        
        # Add user-provided conditions (original values, keep sex as string)
        result.insert(0, 'AGE', age)
        result.insert(1, 'SEX', sex)
        result.insert(2, 'BMXHT', height)
        result.insert(3, 'BMXWT', weight)
        
        return result
    
    def validate_generation(self, age, sex, height, weight, n=100, seed=42):
        """
        Validate that generation works properly
        
        Checks:
        1. Features are in reasonable ranges
        2. Different patients are actually different
        3. Generated distributions match real data
        
        Args:
            age, sex, height, weight: User inputs to test
            n: Number of patients to generate for testing
            seed: Random seed
        
        Returns:
            Dictionary with validation results
        """
        print("\n" + "="*70)
        print("VALIDATING GENERATOR")
        print("="*70)
        
        # Generate patients
        generated = self.generate(age, sex, height, weight, n=n, seed=seed)
        
        # Check 1: Feature ranges
        print("\n[CHECK 1] Feature ranges:")
        for col in BODY_MEASUREMENT_COLS[:3]:
            gen_mean = generated[col].mean()
            gen_std = generated[col].std()
            real_mean = self.df[col].mean()
            real_std = self.df[col].std()
            print(f"  {col}: Gen={gen_mean:.2f}±{gen_std:.2f}, "
                  f"Real={real_mean:.2f}±{real_std:.2f}")
        
        # Check 2: Questionnaire values are discrete
        print("\n[CHECK 2] Questionnaire responses (should be 1, 2, or 9):")
        for col in QUESTIONNAIRE_COLS[:3]:
            unique_vals = sorted(generated[col].unique())
            print(f"  {col}: {unique_vals}")
        
        # Check 3: Variety (all patients should be different)
        print("\n[CHECK 3] Patient variety:")
        duplicates = generated.duplicated().sum()
        print(f"  Duplicate patients: {duplicates}/{n}")
        if duplicates == 0:
            print("  [OK] All patients are unique!")
        else:
            print("  [!] Some patients are identical - might need more training")
        
        # Check 4: BMI calculation
        print("\n[CHECK 4] BMI validation:")
        expected_bmi = weight / (height / 100) ** 2
        actual_bmi = generated['BMXBMI'].iloc[0]
        print(f"  Expected BMI: {expected_bmi:.2f}")
        print(f"  Generated BMI: {actual_bmi:.2f}")
        if abs(expected_bmi - actual_bmi) < 0.01:
            print("  [OK] BMI matches height/weight!")
        
        print("="*70 + "\n")
        
        return {
            'duplicates': duplicates,
            'generated_df': generated
        }
