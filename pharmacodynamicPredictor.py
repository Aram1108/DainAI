"""Production-grade pharmacodynamic predictor with cross-attention architecture.

Predicts changes to 22 lab biomarkers given:
- Patient baseline state S₀ (41 features)
- Drug molecular embedding D (768 features)

Architecture: Cross-Attention Predictor
- Drug embedding "attends" to patient features
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
    import torch.nn.functional as F
except Exception as e:
    raise ImportError("PyTorch is required. Install via: pip install torch") from e

from constants import DRUG_EMBED_DIM, DEVICE


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
# CROSS-ATTENTION PREDICTOR
# ============================================================================

class CrossAttentionPredictor(nn.Module):
    """Drug-Patient Cross-Attention for Pharmacodynamic Prediction.
    
    Architecture:
    1. Project drug (768) and patient (41) to same dimension (256)
    2. Drug queries patient features via multi-head cross-attention
    3. Multiple attention layers with residual connections
    4. Predict only 22 lab biomarker changes
    
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
        
        # Project drug and patient to same dimension
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.patient_proj = nn.Sequential(
            nn.Linear(patient_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multi-layer cross-attention: drug attends to patient
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
        
        self.feedforward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Final prediction head for 22 lab biomarkers only
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, lab_biomarker_dim)
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
        # Project to hidden dimension
        patient_h = self.patient_proj(patient_state).unsqueeze(1)  # (B, 1, H)
        drug_h = self.drug_proj(drug_emb).unsqueeze(1)  # (B, 1, H)
        
        # Cross-attention layers with residual connections
        attended = drug_h
        all_attn_weights = []
        
        for i, (attn_layer, norm_attn, norm_ff, ff) in enumerate(
            zip(self.cross_attention_layers, self.layer_norms_attn, 
                self.layer_norms_ff, self.feedforward_layers)
        ):
            # Cross-attention: drug queries patient
            attn_out, attn_weights = attn_layer(
                query=attended,
                key=patient_h,
                value=patient_h
            )
            attended = norm_attn(attended + attn_out)  # Residual + norm
            
            # Feedforward with residual
            ff_out = ff(attended)
            attended = norm_ff(attended + ff_out)
            
            if return_attention:
                all_attn_weights.append(attn_weights.detach())
        
        # Predict changes to 22 lab biomarkers
        delta_labs = self.predictor(attended.squeeze(1))
        
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
        
        # Load feature bounds from GAN metadata
        self.feature_bounds = self._load_feature_bounds()
        
        # Build model
        torch.manual_seed(seed)
        
        if predictor_type == 'cross_attention':
            self.model = CrossAttentionPredictor(
                patient_dim=patient_dim,
                drug_dim=drug_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
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
            
            means = metadata['usage']['normalization']['features_mean']
            stds = metadata['usage']['normalization']['features_std']
            
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
            
        except Exception as e:
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
        # Convert to tensors
        patient_state = self._prepare_patient_state(S0_df)
        drug_emb = torch.tensor(
            drug_embedding, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Expand drug embedding to batch size
        batch_size = patient_state.size(0)
        drug_emb = drug_emb.unsqueeze(0).expand(batch_size, -1)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            delta, _ = self.model(patient_state, drug_emb, return_attention=False)
        
        # Apply constraints if enabled
        if self.use_constraints:
            delta = self._apply_constraints(patient_state, delta)
        
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
        # Convert to tensors
        patient_state = self._prepare_patient_state(S0_df)
        drug_emb = torch.tensor(
            drug_embedding,
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
                delta, _ = self.model(patient_state, drug_emb, return_attention=False)
                if self.use_constraints:
                    delta = self._apply_constraints(patient_state, delta)
                samples.append(delta)
        
        # Return to eval mode
        self.model.eval()
        
        # Stack samples and compute statistics
        samples = torch.stack(samples)  # (n_samples, B, 22)
        
        mean = samples.mean(dim=0).cpu().numpy()
        std = samples.std(dim=0).cpu().numpy()
        lower = (samples.mean(dim=0) - 1.96 * samples.std(dim=0)).cpu().numpy()
        upper = (samples.mean(dim=0) + 1.96 * samples.std(dim=0)).cpu().numpy()
        
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
        patient_state = self._prepare_patient_state(S0_df)
        drug_emb = torch.tensor(
            drug_embedding,
            dtype=torch.float32,
            device=self.device
        )
        
        batch_size = patient_state.size(0)
        drug_emb = drug_emb.unsqueeze(0).expand(batch_size, -1)
        
        self.model.eval()
        with torch.no_grad():
            delta, attn_weights = self.model(
                patient_state, 
                drug_emb, 
                return_attention=True
            )
        
        if self.use_constraints:
            delta = self._apply_constraints(patient_state, delta)
        
        delta_df = pd.DataFrame(
            delta.cpu().numpy(),
            columns=self.lab_features,
            index=S0_df.index
        )
        
        # Convert attention weights to numpy
        attn_weights_np = [w.cpu().numpy() for w in attn_weights]
        
        return delta_df, attn_weights_np
    
    def _prepare_patient_state(self, S0_df: pd.DataFrame) -> torch.Tensor:
        """Convert patient DataFrame to tensor, handling missing columns."""
        # Ensure all 41 features are present
        rows = []
        for _, row in S0_df.iterrows():
            vals = []
            for col in self.all_patient_features:
                try:
                    vals.append(float(row[col]) if col in S0_df.columns else 0.0)
                except:
                    vals.append(0.0)
            rows.append(vals)
        
        state_mat = np.vstack(rows).astype(np.float32)
        return torch.tensor(state_mat, dtype=torch.float32, device=self.device)
    
    def _apply_constraints(
        self,
        patient_state: torch.Tensor,
        delta: torch.Tensor
    ) -> torch.Tensor:
        """Apply physiological bounds to predictions."""
        # Extract current lab values (indices 12-33)
        current_labs = patient_state[:, LAB_START_IDX:LAB_END_IDX]
        
        # Compute new lab values
        new_labs = current_labs + delta
        
        # Apply bounds
        for i, feature in enumerate(self.lab_features):
            if feature in self.feature_bounds:
                lower, upper = self.feature_bounds[feature]
                new_labs[:, i] = torch.clamp(
                    new_labs[:, i],
                    min=lower,
                    max=upper
                )
        
        # Recompute constrained delta
        delta_constrained = new_labs - current_labs
        
        return delta_constrained
    
    def save(self, path: str):
        """Save model and metadata."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save({
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
        }, save_path)
        
        # Save metadata
        metadata = {
            'model_info': {
                'name': 'Cross-Attention Pharmacodynamic Predictor',
                'version': '1.0',
                'architecture': 'CrossAttentionPredictor',
                'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'trained': False,
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
        
        metadata_path = save_path.parent / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model saved to: {save_path}")
        print(f"✓ Metadata saved to: {metadata_path}")
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
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
