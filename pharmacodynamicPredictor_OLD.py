"""AI-based pharmacodynamic predictor (PyTorch MLP).

This predictor is a small neural network that maps the concatenation of the
baseline state vector and a drug embedding to a predicted delta vector ΔS.
"""
from typing import Iterable
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except Exception as e:
    raise ImportError("PyTorch is required for the pharmacodynamic predictor. Install via: pip install torch") from e

from constants import PREDICTOR_HIDDEN, DRUG_EMBED_DIM, DEVICE


class _PredictorNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = PREDICTOR_HIDDEN, out_dim: int = None):
        super().__init__()
        out_dim = out_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PharmacodynamicPredictor:
    def __init__(self, feature_names: Iterable[str], embed_dim: int = DRUG_EMBED_DIM, seed: int = 0):
        self.feature_names = list(feature_names)
        self.feature_dim = len(self.feature_names)
        self.embed_dim = embed_dim
        self.device = DEVICE

        # Build network that consumes [state (F), drug_emb (E)] -> delta (F)
        input_dim = self.feature_dim + self.embed_dim
        torch.manual_seed(seed)
        self.model = _PredictorNet(input_dim=input_dim, out_dim=self.feature_dim).to(self.device)
        self.model.eval()

    def predict_delta(self, S0_df: pd.DataFrame, drug_embedding: np.ndarray) -> pd.DataFrame:
        """Predict ΔS for each row in S0_df given a drug_embedding vector.

        Returns a DataFrame of shape (n_rows, feature_dim) with same index as S0_df.
        """
        import torch

        # Ensure embedding length
        e = np.asarray(drug_embedding)
        if e.shape[0] != self.embed_dim:
            raise ValueError('drug_embedding dimension mismatch')

        # Prepare state matrix (rows x features) in numeric order of feature_names
        rows = []
        for _, row in S0_df.iterrows():
            vals = []
            for col in self.feature_names:
                try:
                    vals.append(float(row[col]) if col in S0_df.columns else 0.0)
                except Exception:
                    vals.append(0.0)
            rows.append(vals)

        state_mat = np.vstack(rows).astype(np.float32)

        # tile drug embedding
        emb_tile = np.tile(e.astype(np.float32), (state_mat.shape[0], 1))

        inp = np.concatenate([state_mat, emb_tile], axis=1)
        inp_t = torch.tensor(inp, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            out_t = self.model(inp_t).cpu().numpy()

        df_delta = pd.DataFrame(out_t, columns=self.feature_names, index=S0_df.index)
        return df_delta


if __name__ == '__main__':
    import pandas as pd
    fn = [f'F{i}' for i in range(10)]
    pred = PharmacodynamicPredictor(fn, embed_dim=64)
    s0 = pd.DataFrame([np.arange(10)], columns=fn)
    emb = np.ones(64)
    print(pred.predict_delta(s0, emb))
