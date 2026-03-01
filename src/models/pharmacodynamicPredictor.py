"""Production-grade pharmacodynamic predictor with cross-attention architecture.

Predicts changes to 22 lab biomarkers given:
- Patient baseline state S₀ (41 features)
- Drug molecular embedding D (768 features)

Architecture: Cross-Attention Predictor
- Drug embedding "attends" to patient features (each feature as separate token)
- Identifies which biomarkers will be affected
- Outputs changes ONLY to 22 lab biomarkers (not body measurements or questionnaires)
"""
from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
import json
from pathlib import Path
import warnings

try:
    import torch
    import torch.nn as nn
except Exception as e:
    raise ImportError("PyTorch is required. Install via: pip install torch") from e

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.constants import DRUG_EMBED_DIM, DEVICE


# ============================================================================
# FEATURE CONFIGURATION (41 total patient features)
# ============================================================================

# Feature ordering (0-40):
PATIENT_FEATURES = [
    # Input conditions (0-3): 4 features
    'AGE', 'SEX', 'BMXHT', 'BMXWT',
    
    # Body measurements (4-11): 8 features  
    'BMXARMC', 'BMXARML', 'BMXBMI', 'BMXLEG',
    'BMXSUB', 'BMXTHICR', 'BMXTRI', 'BMXWAIST',
    
    # Lab results (12-33): 22 features ⭐ WE PREDICT THESE
    'LBDSCASI', 'LBDSCH', 'LBDSCR', 'LBDTC',
    'LBXBCD', 'LBXBPB', 'LBXCRP', 'LBXSAL',
    'LBXSAS', 'LBXSBU', 'LBXSCA', 'LBXSCH',
    'LBXSCL', 'LBXSGB', 'LBXSGL', 'LBXSGT',
    'LBXSK', 'LBXSNA', 'LBXSOS', 'LBXSTP',
    'LBXSUA', 'LBXTC',
    
    # Questionnaires (34-40): 7 features
    'MCQ160B', 'MCQ160C', 'MCQ160E', 'MCQ160F',
    'MCQ160K', 'MCQ160L', 'MCQ220'
]

# Only predict lab biomarkers (indices 12-33)
LAB_BIOMARKER_FEATURES = PATIENT_FEATURES[12:34]  # 22 features

# Indices for extraction
LAB_START_IDX = 12
LAB_END_IDX = 34  # Exclusive


# ============================================================================
# CROSS-ATTENTION PREDICTOR (Improved Architecture)
# ============================================================================

class CrossAttentionPredictor(nn.Module):
    """Drug-Patient Cross-Attention for Pharmacodynamic Prediction.
    
    Architecture:
    1. Project drug (768) to hidden_dim (256)
    2. Project each patient feature (41) to hidden_dim (256) - treat as sequence
    3. Drug queries patient features via multi-head cross-attention
    4. Multiple attention layers with residual connections
    5. Predict only 22 lab biomarker changes
    
    This allows the drug embedding to "attend" to specific patient features
    that are relevant for predicting biomarker responses.
    """
    def __init__(
        self,
        patient_dim: int = 41,
        drug_dim: int = DRUG_EMBED_DIM,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        lab_biomarker_dim: int = 22
    ):
        super().__init__()
        
        self.patient_dim = patient_dim
        self.drug_dim = drug_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.lab_biomarker_dim = lab_biomarker_dim
        
        # Project drug to hidden dimension
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Project each patient feature to hidden dimension (treat as sequence)
        # Input: (B, 41) -> Output: (B, 41, hidden_dim)
        self.patient_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Each feature is a scalar
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multi-layer cross-attention: drug attends to patient features
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.layer_norms_attn = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.layer_norms_ff = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Feedforward with moderate expansion (balanced capacity vs regularization)
        expansion_factor = 4  # Standard transformer expansion
        self.feedforward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * expansion_factor),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * expansion_factor, hidden_dim),
                nn.Dropout(dropout * 0.5)  # Slightly less dropout on second layer
            ) for _ in range(num_layers)
        ])
        
        # Prediction head for 22 lab biomarkers with regularization
        # Two-layer head with dropout for better generalization
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),  # Add layer norm for stability
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # Additional layer
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, lab_biomarker_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Xavier/He initialization for better gradient flow."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                if 'norm' not in name.lower():
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        patient_state: torch.Tensor,
        drug_emb: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            patient_state: (B, 41) - full patient features
            drug_emb: (B, 768) - drug embedding
            return_attention: whether to return attention weights for interpretability
            
        Returns:
            delta: (B, 22) - predicted changes to lab biomarkers only
            attn_weights: (optional) list of attention weights from each layer
        """
        batch_size = patient_state.size(0)
        
        # Project drug to hidden dimension: (B, 768) -> (B, 1, H)
        drug_h = self.drug_proj(drug_emb).unsqueeze(1)  # (B, 1, H)
        
        # Project patient features as sequence: (B, 41) -> (B, 41, H)
        # Each feature becomes a token
        patient_state_expanded = patient_state.unsqueeze(-1)  # (B, 41, 1)
        patient_h = self.patient_proj(patient_state_expanded)  # (B, 41, H)
        
        # Cross-attention layers with residual connections
        attended = drug_h  # (B, 1, H)
        all_attn_weights = []
        
        for i, (attn_layer, norm_attn, norm_ff, ff) in enumerate(
            zip(self.cross_attention_layers, self.layer_norms_attn, 
                self.layer_norms_ff, self.feedforward_layers)
        ):
            # Cross-attention: drug queries patient features
            attn_out, attn_weights = attn_layer(
                query=attended,      # (B, 1, H) - drug queries
                key=patient_h,       # (B, 41, H) - patient features as keys
                value=patient_h      # (B, 41, H) - patient features as values
            )
            attended = norm_attn(attended + attn_out)  # Residual + norm
            
            # Feedforward with residual
            ff_out = ff(attended)
            attended = norm_ff(attended + ff_out)
            
            if return_attention:
                all_attn_weights.append(attn_weights.detach())
        
        # Predict changes to 22 lab biomarkers
        delta_labs = self.predictor(attended.squeeze(1))  # (B, 1, H) -> (B, H) -> (B, 22)
        
        if return_attention:
            return delta_labs, all_attn_weights
        return delta_labs, None


# ============================================================================
# PHARMACODYNAMIC PREDICTOR (Main API)
# ============================================================================

class PharmacodynamicPredictor:
    """Production-grade pharmacodynamic predictor.
    
    Predicts changes to 22 lab biomarkers given patient state and drug embedding.
    Uses cross-attention architecture to identify drug-patient interactions.
    
    Usage:
        predictor = PharmacodynamicPredictor()
        delta_df = predictor.predict_delta(patient_df, drug_embedding)
        # delta_df has 22 columns (lab biomarkers only)
    """
    
    def __init__(
        self,
        predictor_type: str = 'cross_attention',
        patient_dim: int = 41,
        drug_dim: int = DRUG_EMBED_DIM,
        lab_dim: int = 22,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_constraints: bool = True,
        device: str = DEVICE,
        seed: int = 42
    ):
        """
        Args:
            predictor_type: Type of predictor ('cross_attention')
            patient_dim: Patient state dimension (41 features)
            drug_dim: Drug embedding dimension (768 from hybrid encoder)
            lab_dim: Number of lab biomarkers to predict (22)
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            num_layers: Number of attention layers
            use_constraints: Whether to enforce physiological bounds
            device: Device to run on ('cpu', 'cuda', 'mps')
            seed: Random seed for reproducibility
        """
        self.predictor_type = predictor_type
        self.patient_dim = patient_dim
        self.drug_dim = drug_dim
        self.lab_dim = lab_dim
        self.device = device
        self.use_constraints = use_constraints
        
        # Lab biomarker feature names (22 features, indices 12-33)
        self.lab_features = LAB_BIOMARKER_FEATURES
        self.all_patient_features = PATIENT_FEATURES
        
        # Normalization scalers (will be fitted on training data)
        self.feature_scaler = None  # StandardScaler for patient features (41 features)
        self.lab_scaler = None      # StandardScaler for lab biomarkers (22 features)
        self.drug_scaler = None     # StandardScaler for drug embeddings (768 features)
        
        # Load feature bounds from GAN metadata
        self.feature_bounds = self._load_feature_bounds()
        
        # Build model
        torch.manual_seed(seed)
        if torch.cuda.is_available() and device == 'cuda':
            torch.cuda.manual_seed_all(seed)
        
        if predictor_type == 'cross_attention':
            self.model = CrossAttentionPredictor(
                patient_dim=patient_dim,
                drug_dim=drug_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                lab_biomarker_dim=lab_dim
            ).to(device)
        else:
            raise ValueError(f"Unknown predictor_type: {predictor_type}")
        
        self.model.eval()
        
        print(f"✓ Initialized {predictor_type} predictor")
        print(f"  Parameters: {self.num_parameters:,}")
        print(f"  Input: Patient state (41) + Drug embedding ({drug_dim})")
        print(f"  Output: Lab biomarker changes (22)")
    
    def _load_feature_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Load feature statistics from GAN metadata and compute bounds."""
        metadata_path = Path('models/generator/metadata.json')
        
        if not metadata_path.exists():
            warnings.warn("GAN metadata not found, using default bounds")
            return self._get_default_bounds()
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            means = metadata.get('usage', {}).get('normalization', {}).get('features_mean', {})
            stds = metadata.get('usage', {}).get('normalization', {}).get('features_std', {})
            
            bounds = {}
            for feature in self.lab_features:
                if feature in means and feature in stds:
                    mean = means[feature]
                    std = stds[feature]
                    
                    # Safety bounds: mean ± 4*std (covers ~99.99% of distribution)
                    lower = max(0, mean - 4 * std)
                    upper = mean + 4 * std
                    
                    bounds[feature] = (lower, upper)
            
            # Critical overrides with known physiological limits
            bounds.update({
                'LBXSGL': (50, 300),      # Glucose: strict clinical limits
                'LBXSCA': (7.0, 12.0),    # Calcium: narrow safe range
                'LBXSK': (2.5, 6.5),      # Potassium: critical for heart
                'LBXSNA': (130, 150),     # Sodium: critical for neurons
                'LBXSCL': (95, 110),      # Chloride: electrolyte balance
                'LBXSCH': (100, 400),     # Cholesterol: clinical range
                'LBXTC': (100, 400),      # Total cholesterol
                'LBXCRP': (0, 10),        # CRP: inflammation marker
            })
            
            return bounds
            
        except (KeyError, json.JSONDecodeError, IOError) as e:
            warnings.warn(f"Failed to load GAN metadata: {e}. Using defaults.")
            return self._get_default_bounds()
    
    def _get_default_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Default physiological bounds if metadata unavailable."""
        return {
            'LBXSGL': (50, 300),      # Glucose
            'LBXSCA': (7.0, 12.0),    # Calcium
            'LBXSK': (2.5, 6.5),      # Potassium
            'LBXSNA': (130, 150),     # Sodium
            'LBXSCL': (95, 110),      # Chloride
            'LBXSCH': (100, 400),     # Cholesterol
            'LBXTC': (100, 400),      # Total cholesterol
            'LBXCRP': (0, 10),        # CRP
            'LBXSAS': (5, 200),       # AST
            'LBXSGT': (5, 300),       # GGT
            'LBXSBU': (5, 50),        # BUN
        }
    
    def _validate_inputs(
        self,
        S0_df: pd.DataFrame,
        drug_embedding: np.ndarray
    ) -> None:
        """Validate input shapes and types."""
        if S0_df.empty:
            raise ValueError("Patient DataFrame is empty")
        
        if not isinstance(drug_embedding, np.ndarray):
            raise TypeError(f"drug_embedding must be numpy array, got {type(drug_embedding)}")
        
        if drug_embedding.ndim != 1:
            raise ValueError(f"drug_embedding must be 1D array, got shape {drug_embedding.shape}")
        
        if drug_embedding.shape[0] != self.drug_dim:
            raise ValueError(
                f"drug_embedding dimension mismatch: expected {self.drug_dim}, "
                f"got {drug_embedding.shape[0]}"
            )
        
        # Check for required patient features
        missing_features = set(self.all_patient_features) - set(S0_df.columns)
        if missing_features:
            warnings.warn(
                f"Missing patient features: {missing_features}. "
                "Will be filled with 0.0"
            )
    
    def predict_delta(
        self,
        S0_df: pd.DataFrame,
        drug_embedding: np.ndarray
    ) -> pd.DataFrame:
        """Predict changes to lab biomarkers (fast, no uncertainty).
        
        Args:
            S0_df: DataFrame with 41 patient features (single row or batch)
            drug_embedding: (768,) drug embedding vector
            
        Returns:
            DataFrame with 22 lab biomarker changes (ΔS)
            Columns: LBDSCASI, LBDSCH, ..., LBXTC (22 features)
        """
        # Validate inputs
        self._validate_inputs(S0_df, drug_embedding)
        
        # Convert to tensors
        patient_state = self._prepare_patient_state(S0_df)
        
        # Normalize drug embedding if scaler is available
        drug_emb_np = drug_embedding.reshape(1, -1)
        if self.drug_scaler is not None:
            drug_emb_np = self.drug_scaler.transform(drug_emb_np)
        drug_emb = torch.tensor(
            drug_emb_np.flatten(),
            dtype=torch.float32, 
            device=self.device
        )
        
        # Expand drug embedding to batch size
        batch_size = patient_state.size(0)
        drug_emb = drug_emb.unsqueeze(0).expand(batch_size, -1)
        
        # Predict (model outputs normalized deltas)
        self.model.eval()
        with torch.no_grad():
            delta_normalized, _ = self.model(patient_state, drug_emb, return_attention=False)
        
        # Denormalize predictions
        if self.lab_scaler is not None:
            delta_np = delta_normalized.cpu().numpy()
            delta_np = self.lab_scaler.inverse_transform(delta_np)
            delta = torch.tensor(delta_np, dtype=torch.float32, device=self.device)
        else:
            delta = delta_normalized
        
        # Apply constraints if enabled (on denormalized values)
        if self.use_constraints:
            patient_state_original = self._get_original_patient_state(S0_df)
            delta = self._apply_constraints(patient_state_original, delta)
        
        # Convert to DataFrame
        delta_df = pd.DataFrame(
            delta.cpu().numpy(),
            columns=self.lab_features,
            index=S0_df.index
        )
        
        return delta_df
    
    def predict_with_uncertainty(
        self,
        S0_df: pd.DataFrame,
        drug_embedding: np.ndarray,
        n_samples: int = 50
    ) -> Dict[str, pd.DataFrame]:
        """Predict with uncertainty estimates via Monte Carlo Dropout.
        
        Args:
            S0_df: DataFrame with 41 patient features
            drug_embedding: (768,) drug embedding vector
            n_samples: Number of MC samples for uncertainty estimation
            
        Returns:
            Dictionary with 'mean', 'std', 'lower_95', 'upper_95' DataFrames
        """
        # Validate inputs
        self._validate_inputs(S0_df, drug_embedding)
        
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        
        # Convert to tensors
        patient_state = self._prepare_patient_state(S0_df)
        
        # Normalize drug embedding if scaler is available
        drug_emb_np = drug_embedding.reshape(1, -1)
        if self.drug_scaler is not None:
            drug_emb_np = self.drug_scaler.transform(drug_emb_np)
        drug_emb = torch.tensor(
            drug_emb_np.flatten(),
            dtype=torch.float32,
            device=self.device
        )
        
        batch_size = patient_state.size(0)
        drug_emb = drug_emb.unsqueeze(0).expand(batch_size, -1)
        
        # Enable dropout for uncertainty
        self.model.train()
        
        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                delta_normalized, _ = self.model(patient_state, drug_emb, return_attention=False)
                # Denormalize
                if self.lab_scaler is not None:
                    delta_np = delta_normalized.cpu().numpy()
                    delta_np = self.lab_scaler.inverse_transform(delta_np)
                    delta = torch.tensor(delta_np, dtype=torch.float32, device=self.device)
                else:
                    delta = delta_normalized
                if self.use_constraints:
                    patient_state_original = self._get_original_patient_state(S0_df)
                    delta = self._apply_constraints(patient_state_original, delta)
                samples.append(delta)
        
        # Return to eval mode
        self.model.eval()
        
        # Stack samples and compute statistics (vectorized)
        samples = torch.stack(samples)  # (n_samples, B, 22)
        
        mean = samples.mean(dim=0).cpu().numpy()
        std = samples.std(dim=0).cpu().numpy()
        lower = (mean - 1.96 * std)
        upper = (mean + 1.96 * std)
        
        return {
            'mean': pd.DataFrame(mean, columns=self.lab_features, index=S0_df.index),
            'std': pd.DataFrame(std, columns=self.lab_features, index=S0_df.index),
            'lower_95': pd.DataFrame(lower, columns=self.lab_features, index=S0_df.index),
            'upper_95': pd.DataFrame(upper, columns=self.lab_features, index=S0_df.index)
        }
    
    def predict_with_attention(
        self,
        S0_df: pd.DataFrame,
        drug_embedding: np.ndarray
    ) -> Tuple[pd.DataFrame, List[np.ndarray]]:
        """Predict with attention weights for interpretability.
        
        Returns:
            delta_df: Predicted changes (22 features)
            attention_weights: List of attention weight matrices from each layer
        """
        # Validate inputs
        self._validate_inputs(S0_df, drug_embedding)
        
        patient_state = self._prepare_patient_state(S0_df)
        
        # Normalize drug embedding if scaler is available
        drug_emb_np = drug_embedding.reshape(1, -1)
        if self.drug_scaler is not None:
            drug_emb_np = self.drug_scaler.transform(drug_emb_np)
        drug_emb = torch.tensor(
            drug_emb_np.flatten(),
            dtype=torch.float32,
            device=self.device
        )
        
        batch_size = patient_state.size(0)
        drug_emb = drug_emb.unsqueeze(0).expand(batch_size, -1)
        
        self.model.eval()
        with torch.no_grad():
            delta_normalized, attn_weights = self.model(
                patient_state, 
                drug_emb, 
                return_attention=True
            )
        
        # Denormalize predictions
        if self.lab_scaler is not None:
            delta_np = delta_normalized.cpu().numpy()
            delta_np = self.lab_scaler.inverse_transform(delta_np)
            delta = torch.tensor(delta_np, dtype=torch.float32, device=self.device)
        else:
            delta = delta_normalized
        
        # Apply constraints if enabled (need original-scale patient_state)
        if self.use_constraints:
            patient_state_original = self._get_original_patient_state(S0_df)
            delta = self._apply_constraints(patient_state_original, delta)
        
        delta_df = pd.DataFrame(
            delta.cpu().numpy(),
            columns=self.lab_features,
            index=S0_df.index
        )
        
        # Convert attention weights to numpy
        attn_weights_np = [w.cpu().numpy() for w in attn_weights]
        
        return delta_df, attn_weights_np
    
    def _prepare_patient_state(self, S0_df: pd.DataFrame) -> torch.Tensor:
        """Convert patient DataFrame to tensor, handling missing columns and normalizing."""
        # Ensure all 41 features are present
        rows = []
        for _, row in S0_df.iterrows():
            vals = []
            for col in self.all_patient_features:
                try:
                    if col in S0_df.columns:
                        val = float(row[col])
                        if pd.isna(val):
                            val = 0.0
                    else:
                        val = 0.0
                    vals.append(val)
                except (ValueError, TypeError):
                    vals.append(0.0)
            rows.append(vals)
        
        state_mat = np.vstack(rows).astype(np.float32)
        
        if state_mat.shape[1] != self.patient_dim:
            raise ValueError(
                f"Patient state dimension mismatch: expected {self.patient_dim}, "
                f"got {state_mat.shape[1]}"
            )
        
        # Normalize patient features if scaler is available
        if self.feature_scaler is not None:
            state_mat = self.feature_scaler.transform(state_mat)
        
        return torch.tensor(state_mat, dtype=torch.float32, device=self.device)
    
    def _get_original_patient_state(self, S0_df: pd.DataFrame) -> torch.Tensor:
        """Extract original (unnormalized) patient state from DataFrame."""
        rows = []
        for _, row in S0_df.iterrows():
            vals = []
            for col in self.all_patient_features:
                try:
                    if col in S0_df.columns:
                        val = float(row[col])
                        if pd.isna(val):
                            val = 0.0
                    else:
                        val = 0.0
                    vals.append(val)
                except (ValueError, TypeError):
                    vals.append(0.0)
            rows.append(vals)
        
        state_mat = np.vstack(rows).astype(np.float32)
        return torch.tensor(state_mat, dtype=torch.float32, device=self.device)
    
    def _apply_constraints(
        self,
        patient_state: torch.Tensor,
        delta: torch.Tensor
    ) -> torch.Tensor:
        """Apply physiological bounds to predictions (vectorized)."""
        # Extract current lab values (indices 12-33)
        current_labs = patient_state[:, LAB_START_IDX:LAB_END_IDX]
        
        # Compute new lab values
        new_labs = current_labs + delta
        
        # Vectorized constraint application
        # Create bounds tensors: (22,) for lower and upper bounds
        lower_bounds = torch.full(
            (self.lab_dim,),
            fill_value=float('-inf'),
            device=delta.device,
            dtype=delta.dtype
        )
        upper_bounds = torch.full(
            (self.lab_dim,),
            fill_value=float('inf'),
            device=delta.device,
            dtype=delta.dtype
        )
        
        for i, feature in enumerate(self.lab_features):
            if feature in self.feature_bounds:
                lower, upper = self.feature_bounds[feature]
                lower_bounds[i] = lower
                upper_bounds[i] = upper
        
        # Apply bounds vectorized (broadcast automatically)
        new_labs = torch.clamp(
            new_labs,
            min=lower_bounds.unsqueeze(0),
            max=upper_bounds.unsqueeze(0)
        )
        
        # Recompute constrained delta
        delta_constrained = new_labs - current_labs
        
        return delta_constrained
    
    def fit_scalers(self, train_patient_data: np.ndarray, train_lab_data: np.ndarray, 
                    train_drug_embeddings: np.ndarray):
        """Fit normalization scalers on training data. MUST call before training.
        
        Args:
            train_patient_data: (N, 41) array of patient features
            train_lab_data: (N, 22) array of lab biomarker values (baselines or deltas)
            train_drug_embeddings: (N, 768) array of drug embeddings
        """
        from sklearn.preprocessing import StandardScaler
        
        # Fit patient feature scaler
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(train_patient_data)
        
        # Fit lab biomarker scaler
        self.lab_scaler = StandardScaler()
        self.lab_scaler.fit(train_lab_data)
        
        # Fit drug embedding scaler
        self.drug_scaler = StandardScaler()
        self.drug_scaler.fit(train_drug_embeddings)
        
        print(f"✓ Fitted normalization scalers:")
        print(f"  Patient features: mean={self.feature_scaler.mean_[:3]}, std={self.feature_scaler.scale_[:3]}")
        print(f"  Lab biomarkers: mean={self.lab_scaler.mean_[:3]}, std={self.lab_scaler.scale_[:3]}")
        print(f"  Drug embeddings: mean={self.drug_scaler.mean_[:3]}, std={self.drug_scaler.scale_[:3]}")
    
    def save(self, path: str, trained: bool = False, training_metrics: Optional[dict] = None):
        """Save model and metadata.
        
        Args:
            path: Path for the .pt checkpoint (metadata.json is saved in the same directory).
            trained: If True, metadata will record that this is a trained model.
            training_metrics: Optional dict of metrics to store in metadata (e.g. val_loss, val_r2, best_epoch).
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model weights and scalers
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'predictor_type': self.predictor_type,
            'config': {
                'patient_dim': self.patient_dim,
                'drug_dim': self.drug_dim,
                'lab_dim': self.lab_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_heads': self.model.num_heads,
                'num_layers': self.model.num_layers,
            }
        }
        
        # Save scalers if they exist
        import pickle
        if self.feature_scaler is not None:
            checkpoint['feature_scaler'] = pickle.dumps(self.feature_scaler)
        if self.lab_scaler is not None:
            checkpoint['lab_scaler'] = pickle.dumps(self.lab_scaler)
        if self.drug_scaler is not None:
            checkpoint['drug_scaler'] = pickle.dumps(self.drug_scaler)
        
        torch.save(checkpoint, save_path)
        
        # Save metadata
        now = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        metadata = {
            'model_info': {
                'name': 'Cross-Attention Pharmacodynamic Predictor',
                'version': '2.0',
                'architecture': 'CrossAttentionPredictor',
                'created_at': now,
                'trained': trained,
                'initialization': 'xavier_uniform'
            },
            'architecture_config': {
                'patient_dim': self.patient_dim,
                'drug_dim': self.drug_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_heads': self.model.num_heads,
                'num_layers': self.model.num_layers,
                'output_dim': self.lab_dim
            },
            'parameters': {
                'total': self.num_parameters,
                'trainable': self.num_parameters
            },
            'input_features': {
                'patient_state': self.all_patient_features,
                'drug_embedding_dim': self.drug_dim
            },
            'output_features': self.lab_features,
            'feature_bounds': {k: list(v) for k, v in self.feature_bounds.items()}
        }
        if trained:
            metadata['model_info']['saved_at'] = now
        if training_metrics:
            metadata['training_metrics'] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in training_metrics.items()}
        
        metadata_path = save_path.parent / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model saved to: {save_path}")
        print(f"✓ Metadata saved to: {metadata_path}")
    
    def load(self, path: str):
        """Load model weights with validation."""
        checkpoint_path = Path(path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            raise IOError(f"Failed to load checkpoint: {e}") from e
        
        # Validate checkpoint structure
        if 'model_state_dict' not in checkpoint:
            raise ValueError("Checkpoint missing 'model_state_dict' key")
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            # Validate dimensions match
            if config.get('patient_dim') != self.patient_dim:
                raise ValueError(
                    f"Patient dimension mismatch: checkpoint has {config.get('patient_dim')}, "
                    f"model expects {self.patient_dim}"
                )
            if config.get('drug_dim') != self.drug_dim:
                raise ValueError(
                    f"Drug dimension mismatch: checkpoint has {config.get('drug_dim')}, "
                    f"model expects {self.drug_dim}"
                )
            if config.get('lab_dim') != self.lab_dim:
                raise ValueError(
                    f"Lab dimension mismatch: checkpoint has {config.get('lab_dim')}, "
                    f"model expects {self.lab_dim}"
                )
        
        # Load state dict
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load model state dict. Model architecture may have changed. {e}"
            ) from e
        
        # Load scalers if they exist (for backward compatibility with old checkpoints)
        if 'feature_scaler' in checkpoint or 'lab_scaler' in checkpoint or 'drug_scaler' in checkpoint:
            import pickle
            if 'feature_scaler' in checkpoint:
                self.feature_scaler = pickle.loads(checkpoint['feature_scaler'])
            if 'lab_scaler' in checkpoint:
                self.lab_scaler = pickle.loads(checkpoint['lab_scaler'])
            if 'drug_scaler' in checkpoint:
                self.drug_scaler = pickle.loads(checkpoint['drug_scaler'])
            print("  ✓ Loaded normalization scalers from checkpoint")
        
        print(f"✓ Model loaded from: {path}")
    
    @property
    def num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ============================================================================
# DEMO
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("PHARMACODYNAMIC PREDICTOR DEMO")
    print("=" * 70)
    
    # Initialize predictor
    print("\n1. Initializing Cross-Attention Predictor...")
    predictor = PharmacodynamicPredictor(
        predictor_type='cross_attention',
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        use_constraints=True
    )
    
    # Create dummy patient data (41 features)
    print("\n2. Creating dummy patient...")
    patient_data = {
        # Input conditions
        'AGE': [45], 'SEX': [1], 'BMXHT': [175.0], 'BMXWT': [80.0],
        # Body measurements (8)
        'BMXARMC': [30.0], 'BMXARML': [36.0], 'BMXBMI': [26.1], 'BMXLEG': [40.0],
        'BMXSUB': [18.0], 'BMXTHICR': [52.0], 'BMXTRI': [18.0], 'BMXWAIST': [92.0],
        # Labs (22)
        'LBDSCASI': [2.38], 'LBDSCH': [4.87], 'LBDSCR': [71.2], 'LBDTC': [4.91],
        'LBXBCD': [0.48], 'LBXBPB': [1.88], 'LBXCRP': [0.36], 'LBXSAL': [4.33],
        'LBXSAS': [24.2], 'LBXSBU': [12.6], 'LBXSCA': [9.51], 'LBXSCH': [188.2],
        'LBXSCL': [102.9], 'LBXSGB': [3.06], 'LBXSGL': [93.2], 'LBXSGT': [24.9],
        'LBXSK': [4.05], 'LBXSNA': [139.0], 'LBXSOS': [277.5], 'LBXSTP': [7.38],
        'LBXSUA': [5.24], 'LBXTC': [189.7],
        # Questionnaires (7)
        'MCQ160B': [2.0], 'MCQ160C': [2.0], 'MCQ160E': [2.0], 'MCQ160F': [2.0],
        'MCQ160K': [2.0], 'MCQ160L': [2.0], 'MCQ220': [2.0]
    }
    patient_df = pd.DataFrame(patient_data)
    
    # Create dummy drug embedding (768 dimensions)
    print("3. Creating dummy drug embedding...")
    drug_emb = np.random.randn(768).astype(np.float32)
    
    # Test prediction
    print("\n4. Testing prediction...")
    delta_df = predictor.predict_delta(patient_df, drug_emb)
    print(f"\n   Predicted changes to {len(delta_df.columns)} lab biomarkers:")
    print(delta_df.head())
    
    # Test with uncertainty
    print("\n5. Testing prediction with uncertainty...")
    uncertainty = predictor.predict_with_uncertainty(patient_df, drug_emb, n_samples=30)
    print(f"\n   Mean prediction (first 5 labs):")
    print(uncertainty['mean'].iloc[:, :5])
    print(f"\n   Uncertainty (std, first 5 labs):")
    print(uncertainty['std'].iloc[:, :5])
    
    # Test with attention
    print("\n6. Testing prediction with attention weights...")
    delta_df, attn_weights = predictor.predict_with_attention(patient_df, drug_emb)
    print(f"\n   Attention weights from {len(attn_weights)} layers")
    print(f"   Layer 0 shape: {attn_weights[0].shape}")
    
    # Save model
    print("\n7. Saving model...")
    save_dir = Path('models/pharmacodynamic_predictor')
    save_dir.mkdir(parents=True, exist_ok=True)
    predictor.save(save_dir / 'predictor.pt')
    
    print("\n" + "=" * 70)
    print("✓ Demo complete!")
    print("=" * 70)
