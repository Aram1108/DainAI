"""Temporal dynamics simulator that consumes S0, drug embedding and predictor.

Simulates repeated application of the pharmacodynamic predictor at fixed
time intervals with feature-specific time constants and stochastic noise
for realistic dynamics. Returns a time-series DataFrame of states.

Updated to handle partial state updates:
- Predictor returns changes to 22 lab biomarkers only
- Simulator mutates ONLY lab features (indices 12-33)
- Body measurements and questionnaires remain frozen
"""
import pandas as pd
import numpy as np
from typing import List

# Lab biomarker feature names (22 features that get updated)
LAB_BIOMARKER_FEATURES = [
    'LBDSCASI', 'LBDSCH', 'LBDSCR', 'LBDTC',
    'LBXBCD', 'LBXBPB', 'LBXCRP', 'LBXSAL',
    'LBXSAS', 'LBXSBU', 'LBXSCA', 'LBXSCH',
    'LBXSCL', 'LBXSGB', 'LBXSGL', 'LBXSGT',
    'LBXSK', 'LBXSNA', 'LBXSOS', 'LBXSTP',
    'LBXSUA', 'LBXTC'
]


def simulate(initial_state: pd.DataFrame,
             drug_embedding: np.ndarray,
             predictor,
             steps: int = 12,
             dt_min: int = 5,
             noise_level: float = 0.01) -> pd.DataFrame:
    """Simulate `steps` timesteps of dt_min minutes starting from initial_state.

    Parameters:
    -----------
    initial_state: DataFrame with a single row (baseline state with 41 features)
    predictor: object with method predict_delta(S_df, drug_embedding)
                Returns DataFrame with 22 lab biomarker changes
    steps: int - number of steps to simulate
    dt_min: int - time delta in minutes
    noise_level: float - amplitude of stochastic noise (0 = no noise)
    
    Returns:
    --------
    DataFrame with index = minutes and columns = all 41 features
    
    Note: Only lab biomarkers (22 features) are mutated. 
          Body measurements and questionnaires remain constant.
    """
    if initial_state.shape[0] != 1:
        raise ValueError('initial_state must have a single row')

    # Compute per-feature time constants (in steps) for realistic peak dynamics
    # Faster features peak earlier, slower features peak later
    n_lab_features = len(LAB_BIOMARKER_FEATURES)
    rng = np.random.RandomState(0)
    time_constants = rng.uniform(2, 8, size=n_lab_features)  # peaks between step 2-8
    
    current = initial_state.copy()
    records = []
    time_points = []

    for t in range(steps):
        minute = t * dt_min
        time_points.append(minute)
        records.append(current.iloc[0].to_dict())

        # Predict delta for lab biomarkers only (returns 22-column DataFrame)
        delta_labs = predictor.predict_delta(current, drug_embedding)
        
        # Apply feature-specific time constants: modulate delta by (1 - exp(-t/tau))
        # This causes features to peak at different times
        delta_vals = delta_labs.iloc[0].values.astype(float)
        for i in range(len(delta_vals)):
            tau = time_constants[i]
            # Decay factor: features reach peak gradually
            decay = 1.0 - np.exp(-(t + 1) / tau)
            delta_vals[i] *= decay
        
        # Add small stochastic noise for realism (optional)
        if noise_level > 0:
            noise = rng.normal(0, noise_level, size=delta_vals.shape)
            delta_vals += noise
        
        # Apply changes ONLY to lab biomarkers (freeze other features)
        for i, lab_feature in enumerate(LAB_BIOMARKER_FEATURES):
            if lab_feature in current.columns:
                current.loc[current.index[0], lab_feature] += delta_vals[i]
        
        # Body measurements (BMXARMC, BMXBMI, etc.) stay frozen
        # Input conditions (AGE, SEX, BMXHT, BMXWT) stay frozen
        # Questionnaires (MCQ160B, etc.) stay frozen

    # final record at end time
    minute = steps * dt_min
    time_points.append(minute)
    records.append(current.iloc[0].to_dict())

    ts = pd.DataFrame(records, index=time_points)
    ts.index.name = 'minutes'
    return ts


if __name__ == '__main__':
    import pandas as pd
    from pharmacodynamicPredictor import PharmacodynamicPredictor
    fn = [f'F{i}' for i in range(5)]
    s0 = pd.DataFrame([[1,2,3,4,5]], columns=fn)
    pred = PharmacodynamicPredictor(fn, embed_dim=32)
    emb = np.ones(32)
    print(simulate(s0, emb, pred, steps=3, dt_min=5))
