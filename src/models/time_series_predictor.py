"""Time Series Predictor: Converts static drug response predictions into temporal trajectories.

IMPORTANT: This model ONLY converts static predictions (from PharmacodynamicPredictor) into 
time series trajectories. It does NOT predict drug effects - that's done by PharmacodynamicPredictor.

Pipeline:
1. PharmacodynamicPredictor predicts static changes (baseline + final_delta)
2. TimeSeriesPredictor converts those static predictions into temporal trajectories
   showing how delta evolves over time (10, 20, 30, 60, 90, 180 days)

Uses Transformer architecture to learn trajectory shapes from:
- Static final_delta (from PharmacodynamicPredictor)
- Drug-specific response profiles (statins vs. metformin respond differently)
- Patient adherence (non-adherent patients have slower/less complete responses)
- Multi-metric joint prediction (all lab values simultaneously)
- Uncertainty quantification (confidence intervals)

This model is MANDATORY - main.py will raise an error if it's not trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TimeSeriesConfig:
    """Configuration for time series predictor."""
    timepoints: List[int] = None  # [10, 20, 30, 60, 90, 180] days
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    max_delta_magnitude: float = 200.0  # For normalization
    
    def __post_init__(self):
        if self.timepoints is None:
            self.timepoints = [10, 20, 30, 60, 90, 180]


class DrugEmbedding(nn.Module):
    """Embed drug names into learned representations."""
    
    def __init__(self, drug_vocab_size: int, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(drug_vocab_size, embed_dim)
        self.embed_dim = embed_dim
    
    def forward(self, drug_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(drug_ids)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timepoints."""
    
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TimeSeriesTransformer(nn.Module):
    """Transformer-based model for predicting temporal trajectories.
    
    Architecture:
    1. Input encoding: Patient features + drug embedding + baseline + final delta
    2. Timepoint embeddings: Learnable embeddings for each timepoint
    3. Transformer encoder: Self-attention over timepoints
    4. Output projection: Predict delta at each timepoint
    """
    
    def __init__(
        self,
        patient_feature_dim: int,
        drug_vocab_size: int,
        num_metrics: int,
        config: TimeSeriesConfig,
        device: str = 'cpu'
    ):
        super().__init__()
        self.config = config
        self.num_metrics = num_metrics
        self.num_timepoints = len(config.timepoints)
        self.device = device
        
        # Input feature dimensions
        drug_embed_dim = 64
        self.drug_embedding = DrugEmbedding(drug_vocab_size, drug_embed_dim)
        
        # Input: patient features + drug embedding + baseline + final_delta
        input_dim = patient_feature_dim + drug_embed_dim + num_metrics * 2  # baseline + delta
        d_model = config.hidden_dim
        
        # Project input to model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Timepoint embeddings (learnable)
        self.timepoint_embedding = nn.Embedding(self.num_timepoints, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.num_timepoints)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.num_heads,
            dim_feedforward=d_model * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Output projection: predict delta for each metric at each timepoint
        self.output_projection = nn.Linear(d_model, num_metrics)
        
        # Uncertainty estimation (for confidence intervals)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model // 2, num_metrics)  # Predicts log(std)
        )
        
    def forward(
        self,
        patient_features: torch.Tensor,
        drug_ids: torch.Tensor,
        baselines: torch.Tensor,  # (batch, num_metrics)
        final_deltas: torch.Tensor,  # (batch, num_metrics)
        return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            patient_features: (batch, patient_feature_dim)
            drug_ids: (batch,)
            baselines: (batch, num_metrics)
            final_deltas: (batch, num_metrics)
            return_uncertainty: If True, also return uncertainty estimates
        
        Returns:
            predicted_deltas: (batch, num_timepoints, num_metrics)
            uncertainties: (batch, num_timepoints, num_metrics) if return_uncertainty else None
        """
        batch_size = patient_features.size(0)
        
        # Embed drug
        drug_emb = self.drug_embedding(drug_ids)  # (batch, drug_embed_dim)
        
        # Concatenate all input features
        input_features = torch.cat([
            patient_features,
            drug_emb,
            baselines,
            final_deltas
        ], dim=1)  # (batch, input_dim)
        
        # Project to model dimension
        input_emb = self.input_projection(input_features)  # (batch, d_model)
        
        # Expand for each timepoint: (batch, 1, d_model) -> (batch, num_timepoints, d_model)
        input_emb = input_emb.unsqueeze(1).expand(-1, self.num_timepoints, -1)
        
        # Add timepoint embeddings
        timepoint_ids = torch.arange(self.num_timepoints, device=self.device).unsqueeze(0)
        timepoint_emb = self.timepoint_embedding(timepoint_ids)  # (1, num_timepoints, d_model)
        timepoint_emb = timepoint_emb.expand(batch_size, -1, -1)
        
        # Combine input + timepoint embeddings
        x = input_emb + timepoint_emb
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, num_timepoints, d_model)
        
        # Predict deltas
        predicted_deltas = self.output_projection(x)  # (batch, num_timepoints, num_metrics)
        
        # Predict uncertainties if requested
        uncertainties = None
        if return_uncertainty:
            log_std = self.uncertainty_head(x)  # (batch, num_timepoints, num_metrics)
            uncertainties = torch.exp(log_std)  # Convert to std
        
        return predicted_deltas, uncertainties


class TimeSeriesPredictor:
    """High-level interface for time series prediction."""
    
    def __init__(
        self,
        drug_vocab: Dict[str, int],
        metric_names: List[str],
        patient_feature_names: List[str],
        config: Optional[TimeSeriesConfig] = None,
        device: str = 'cpu'
    ):
        """
        Args:
            drug_vocab: Dictionary mapping drug names to integer IDs
            metric_names: List of metric names (e.g., ['LBXTC_delta', 'LBDLDL_delta', ...])
            patient_feature_names: List of patient feature names (e.g., ['age', 'sex', 'bmi', ...])
            config: Configuration object
            device: 'cpu' or 'cuda'
        """
        self.drug_vocab = drug_vocab
        self.metric_names = metric_names
        self.patient_feature_names = patient_feature_names
        self.config = config or TimeSeriesConfig()
        self.device = device
        
        # Initialize model
        self.model = TimeSeriesTransformer(
            patient_feature_dim=len(patient_feature_names),
            drug_vocab_size=len(drug_vocab),
            num_metrics=len(metric_names),
            config=self.config,
            device=device
        ).to(device)
        
        # Inverse drug vocab for lookups
        self.id_to_drug = {v: k for k, v in drug_vocab.items()}
        
    def predict(
        self,
        patient_data: pd.DataFrame,
        return_uncertainty: bool = False,
        drug_embedding: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Convert static predictions (from PharmacodynamicPredictor) into time series trajectories.
        
        This model ONLY converts static predictions to temporal trajectories.
        It does NOT predict drug effects - that's done by PharmacodynamicPredictor.
        
        Args:
            patient_data: DataFrame with columns:
                - patient_id
                - drug_name
                - age, sex, bmi, days_on_drug, adherence
                - {metric}_baseline for each metric (from patient state)
                - {metric}_delta for each metric (FINAL delta from PharmacodynamicPredictor)
            return_uncertainty: If True, include confidence intervals
        
        Returns:
            DataFrame with columns:
                - patient_id
                - metric_name
                - baseline
                - final_delta
                - day_10, day_20, day_30, day_60, day_90, day_180 (delta at each timepoint)
                - If return_uncertainty: also day_10_lower, day_10_upper, etc.
        """
        self.model.eval()
        
        # Prepare inputs
        batch_size = len(patient_data)
        patient_features = []
        drug_ids = []
        baselines = []
        final_deltas = []
        
        for _, row in patient_data.iterrows():
            # Patient features (for trajectory shape modulation)
            pf = [
                row.get('age', 50),
                1.0 if row.get('sex') == 'M' else 0.0,
                row.get('bmi', 25.0),
                row.get('adherence', 1.0),
                row.get('days_on_drug', 90)
            ]
            patient_features.append(pf)
            
            # Drug ID or embedding (for drug-specific trajectory profiles)
            if drug_embedding is not None:
                # Use provided SMILES-based drug embedding
                # Project 768-dim embedding to 64-dim to match model's drug_embed_dim
                if not hasattr(self, '_drug_embedding_projection'):
                    import torch.nn as nn
                    self._drug_embedding_projection = nn.Linear(768, 64).to(self.device)
                    nn.init.xavier_uniform_(self._drug_embedding_projection.weight)
                
                # Project and use as drug embedding (we'll bypass the embedding layer)
                drug_emb_projected = self._drug_embedding_projection(
                    torch.tensor(drug_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
                )
                # Store as a special marker - we'll handle this in the forward pass
                drug_ids.append(-1)  # Special marker for direct embedding
                if not hasattr(self, '_direct_drug_embeddings'):
                    self._direct_drug_embeddings = []
                self._direct_drug_embeddings.append(drug_emb_projected)
            else:
                # Use drug vocabulary (original behavior)
                drug_name = row.get('drug_name', 'Unknown')
                drug_ids.append(self.drug_vocab.get(drug_name, 0))
            
            # Baselines and final deltas (from PharmacodynamicPredictor)
            bl = []
            fd = []
            for metric in self.metric_names:
                # Handle both formats: "LBXTC" or "LBXTC_delta"
                base_name = metric.replace('_delta', '')
                baseline_col = f'{base_name}_baseline'
                delta_col = f'{base_name}_delta' if f'{base_name}_delta' in row else metric
                
                bl.append(row.get(baseline_col, 0.0))
                fd.append(row.get(delta_col, 0.0))
            
            baselines.append(bl)
            final_deltas.append(fd)
        
        # Convert to tensors
        patient_features = torch.tensor(patient_features, dtype=torch.float32).to(self.device)
        baselines = torch.tensor(baselines, dtype=torch.float32).to(self.device)
        final_deltas = torch.tensor(final_deltas, dtype=torch.float32).to(self.device)
        
        # Handle drug embeddings: use direct embeddings if provided, otherwise use drug IDs
        if drug_embedding is not None and hasattr(self, '_direct_drug_embeddings'):
            # Use direct drug embeddings (from SMILES, already projected to 64-dim)
            drug_embeddings = torch.cat(self._direct_drug_embeddings, dim=0)  # (batch, 64)
            
            # Temporarily modify model's forward to use direct embeddings
            original_forward = self.model.forward
            
            def forward_with_direct_embedding(self, patient_features, drug_ids, baselines, final_deltas, return_uncertainty=False):
                batch_size = patient_features.size(0)
                drug_emb = drug_embeddings  # (batch, 64) - use direct embeddings

                # Concatenate all input features
                input_features = torch.cat([
                    patient_features,
                    drug_emb,
                    baselines,
                    final_deltas
                ], dim=1)  # (batch, input_dim)

                # Project to model dimension
                input_emb = self.input_projection(input_features)  # (batch, d_model)

                # Expand for each timepoint
                input_emb = input_emb.unsqueeze(1).expand(-1, self.num_timepoints, -1)

                # Add timepoint embeddings
                timepoint_ids = torch.arange(self.num_timepoints, device=self.device).unsqueeze(0)
                timepoint_emb = self.timepoint_embedding(timepoint_ids)
                timepoint_emb = timepoint_emb.expand(batch_size, -1, -1)

                # Combine input + timepoint embeddings
                x = input_emb + timepoint_emb

                # Add positional encoding
                x = self.pos_encoding(x)

                # Transformer encoding
                x = self.transformer(x)

                # Predict deltas
                predicted_deltas = self.output_projection(x)

                # Predict uncertainties if requested
                uncertainties = None
                if return_uncertainty:
                    log_std = self.uncertainty_head(x)
                    uncertainties = torch.exp(log_std)

                return predicted_deltas, uncertainties
            
            # Temporarily replace forward method
            self.model.forward = forward_with_direct_embedding.__get__(self.model, type(self.model))
            drug_ids = torch.zeros(len(patient_features), dtype=torch.long).to(self.device)  # Dummy, not used
        else:
            # Use drug vocabulary (original behavior)
            drug_ids = torch.tensor(drug_ids, dtype=torch.long).to(self.device)
        
        # Predict trajectory shape (converts static final_delta to time series)
        with torch.no_grad():
            predicted_deltas, uncertainties = self.model(
                patient_features, drug_ids, baselines, final_deltas,
                return_uncertainty=return_uncertainty
            )
        
        # Restore original forward if we modified it
        if drug_embedding is not None and hasattr(self, '_direct_drug_embeddings'):
            self.model.forward = original_forward
            delattr(self, '_direct_drug_embeddings')
        
        # Convert to DataFrame: one row per patient-metric combination
        predicted_deltas = predicted_deltas.cpu().numpy()  # (batch, num_timepoints, num_metrics)
        
        results = []
        for i, (_, row) in enumerate(patient_data.iterrows()):
            patient_id = row.get('patient_id', i)
            drug_name = row.get('drug_name', 'Unknown')
            
            # Create one row per metric
            for metric_idx, metric in enumerate(self.metric_names):
                base_name = metric.replace('_delta', '')
                baseline_col = f'{base_name}_baseline'
                
                result = {
                    'patient_id': patient_id,
                    'metric_name': base_name,  # Store without _delta suffix
                    'baseline': row.get(baseline_col, 0.0),
                    'final_delta': final_deltas[i][metric_idx].item(),
                }
                
                # Add delta at each timepoint
                for t_idx, day in enumerate(self.config.timepoints):
                    delta = predicted_deltas[i, t_idx, metric_idx]
                    result[f'day_{day}'] = float(delta)
                    
                    if return_uncertainty and uncertainties is not None:
                        std = uncertainties[i, t_idx, metric_idx].cpu().numpy()
                        result[f'day_{day}_lower'] = float(delta - 1.96 * std)
                        result[f'day_{day}_upper'] = float(delta + 1.96 * std)
                
                results.append(result)
        
        return pd.DataFrame(results)
    
    def save(self, path: str):
        """Save model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'drug_vocab': self.drug_vocab,
            'metric_names': self.metric_names,
            'patient_feature_names': self.patient_feature_names,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load model from disk.
        
        This will recreate the model with dimensions from the checkpoint,
        so the model must be compatible with the saved checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Extract metadata from checkpoint
        drug_vocab = checkpoint['drug_vocab']
        metric_names = checkpoint['metric_names']
        patient_feature_names = checkpoint['patient_feature_names']
        config = checkpoint['config']
        
        # Recreate model with checkpoint dimensions
        self.drug_vocab = drug_vocab
        self.metric_names = metric_names
        self.patient_feature_names = patient_feature_names
        self.config = config
        
        # Reinitialize model with correct dimensions
        self.model = TimeSeriesTransformer(
            patient_feature_dim=len(patient_feature_names),
            drug_vocab_size=len(drug_vocab),
            num_metrics=len(metric_names),
            config=config,
            device=self.device
        ).to(self.device)
        
        # Now load the state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Update inverse drug vocab
        self.id_to_drug = {v: k for k, v in drug_vocab.items()}

