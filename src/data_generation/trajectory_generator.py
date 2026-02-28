"""Synthetic Trajectory Generator: Creates realistic time series from static predictions.

Generates pharmacologically plausible trajectories using sigmoid/exponential curves
with drug-specific response profiles and adherence modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import curve_fit


# Drug-specific response profiles (from clinical_trials_extractor.py)
DRUG_RESPONSE_PROFILES = {
    'Statin': {
        'tau_days': 30,  # Time to 50% effect
        'steepness': 2.0,  # Sigmoid steepness
        'early_lag_days': 3,  # Days before noticeable effect
    },
    'ACE_Inhibitor': {
        'tau_days': 14,
        'steepness': 3.0,
        'early_lag_days': 1,
    },
    'Metformin': {
        'tau_days': 60,
        'steepness': 1.5,
        'early_lag_days': 7,
    },
    'Diuretic': {
        'tau_days': 7,
        'steepness': 4.0,
        'early_lag_days': 0.5,
    },
    'Antibiotic': {
        'tau_days': 1,
        'steepness': 5.0,
        'early_lag_days': 0,
    },
    'Default': {
        'tau_days': 21,
        'steepness': 2.5,
        'early_lag_days': 2,
    }
}

# Drug class mapping
DRUG_CLASSES = {
    'Simvastatin': 'Statin', 'Atorvastatin': 'Statin', 'Rosuvastatin': 'Statin',
    'Pravastatin': 'Statin', 'Lovastatin': 'Statin',
    'Lisinopril': 'ACE_Inhibitor', 'Enalapril': 'ACE_Inhibitor', 'Ramipril': 'ACE_Inhibitor',
    'Losartan': 'ACE_Inhibitor', 'Valsartan': 'ACE_Inhibitor',
    'Metformin': 'Metformin',
    'Furosemide': 'Diuretic', 'Hydrochlorothiazide': 'Diuretic', 'Chlorthalidone': 'Diuretic',
    'Amoxicillin': 'Antibiotic', 'Azithromycin': 'Antibiotic', 'Ceftriaxone': 'Antibiotic',
    'Gentamicin': 'Antibiotic', 'Ciprofloxacin': 'Antibiotic',
}


def sigmoid_trajectory(
    t: np.ndarray,
    final_delta: float,
    tau: float,
    steepness: float = 2.0,
    early_lag: float = 0.0,
    noise_level: float = 0.05
) -> np.ndarray:
    """
    Generate sigmoid trajectory for drug response.
    
    Formula: delta(t) = final_delta * sigmoid((t - early_lag) / tau, steepness)
    
    Args:
        t: Time points (days)
        final_delta: Final delta value (target)
        tau: Time constant (days to reach ~50% of effect)
        steepness: Sigmoid steepness parameter
        early_lag: Days before effect starts
        noise_level: Relative noise amplitude
    
    Returns:
        Array of delta values at each timepoint
    """
    # Shift time by early lag
    t_shifted = np.maximum(0, t - early_lag)
    
    # Sigmoid function: 1 / (1 + exp(-k * (t/tau - 1)))
    # This gives S-curve that reaches ~50% at t=tau
    k = steepness
    sigmoid_val = 1.0 / (1.0 + np.exp(-k * (t_shifted / tau - 1.0)))
    
    # Scale to final delta
    trajectory = final_delta * sigmoid_val
    
    # Add noise (proportional to magnitude)
    if noise_level > 0:
        noise = np.random.normal(0, abs(final_delta) * noise_level, size=trajectory.shape)
        trajectory += noise
    
    return trajectory


def exponential_trajectory(
    t: np.ndarray,
    final_delta: float,
    tau: float,
    noise_level: float = 0.05
) -> np.ndarray:
    """
    Generate exponential trajectory (alternative to sigmoid).
    
    Formula: delta(t) = final_delta * (1 - exp(-t/tau))
    
    Args:
        t: Time points (days)
        final_delta: Final delta value
        tau: Time constant
        noise_level: Relative noise amplitude
    
    Returns:
        Array of delta values at each timepoint
    """
    trajectory = final_delta * (1.0 - np.exp(-t / tau))
    
    if noise_level > 0:
        noise = np.random.normal(0, abs(final_delta) * noise_level, size=trajectory.shape)
        trajectory += noise
    
    return trajectory


def adherence_modulated_trajectory(
    base_trajectory: np.ndarray,
    adherence: float,
    days_on_drug: int
) -> np.ndarray:
    """
    Modulate trajectory based on adherence.
    
    Non-adherent patients:
    - Slower response (scaled by adherence)
    - More variability
    - May not reach full effect
    
    Args:
        base_trajectory: Base trajectory (full adherence)
        adherence: Adherence rate (0-1)
        days_on_drug: Total days on drug
    
    Returns:
        Modulated trajectory
    """
    # Scale by adherence (linear approximation)
    modulated = base_trajectory * adherence
    
    # Add extra variability for non-adherent patients
    if adherence < 0.9:
        variability = (1.0 - adherence) * 0.1
        noise = np.random.normal(0, variability * np.abs(base_trajectory), size=base_trajectory.shape)
        modulated += noise
    
    return modulated


def generate_trajectory(
    final_delta: float,
    timepoints: List[int],
    drug_name: str,
    adherence: float = 1.0,
    days_on_drug: int = 180,
    metric_type: str = 'default',
    use_sigmoid: bool = True,
    noise_level: float = 0.05,
    patient_variability: float = 0.2
) -> np.ndarray:
    """
    Generate realistic trajectory for a single metric.
    
    Args:
        final_delta: Final delta value from predictor
        timepoints: List of days to predict (e.g., [10, 20, 30, 60, 90, 180])
        drug_name: Name of drug
        adherence: Medication adherence (0-1)
        days_on_drug: Total days on drug
        metric_type: Type of metric ('lipid', 'glucose', 'bp', 'renal', etc.)
        use_sigmoid: If True, use sigmoid; else exponential
        noise_level: Noise amplitude
        patient_variability: Inter-patient variability in response speed
    
    Returns:
        Array of delta values at each timepoint
    """
    # Get drug class and response profile
    drug_class = DRUG_CLASSES.get(drug_name, 'Default')
    profile = DRUG_RESPONSE_PROFILES.get(drug_class, DRUG_RESPONSE_PROFILES['Default'])
    
    # Adjust parameters based on metric type
    tau = profile['tau_days']
    steepness = profile['steepness']
    early_lag = profile['early_lag_days']
    
    # Metric-specific adjustments
    if metric_type == 'glucose':
        tau *= 1.2  # Glucose changes more slowly
    elif metric_type == 'bp':
        tau *= 0.8  # BP changes faster
    elif metric_type == 'renal':
        tau *= 1.5  # Renal markers change slowly
    
    # Add patient variability (some patients respond faster/slower)
    tau *= (1.0 + np.random.normal(0, patient_variability))
    tau = max(1.0, tau)  # Ensure positive
    
    # Generate base trajectory
    t = np.array(timepoints, dtype=float)
    
    if use_sigmoid:
        trajectory = sigmoid_trajectory(
            t, final_delta, tau, steepness, early_lag, noise_level=0
        )
    else:
        trajectory = exponential_trajectory(t, final_delta, tau, noise_level=0)
    
    # Apply adherence modulation
    trajectory = adherence_modulated_trajectory(trajectory, adherence, days_on_drug)
    
    # Add final noise
    if noise_level > 0:
        noise = np.random.normal(0, abs(final_delta) * noise_level, size=trajectory.shape)
        trajectory += noise
    
    # Ensure trajectory doesn't exceed final_delta (for decreasing metrics)
    if final_delta < 0:
        trajectory = np.maximum(trajectory, final_delta)
    else:
        trajectory = np.minimum(trajectory, final_delta)
    
    return trajectory


def generate_training_data(
    predictor_data: pd.DataFrame,
    timepoints: List[int] = [10, 20, 30, 60, 90, 180],
    metric_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate synthetic time series training data from static predictions.
    
    Args:
        predictor_data: DataFrame with columns:
            - patient_id, drug_SMILES (or drug_name), age, sex, bmi, days_on_drug, adherence
            - {metric}_baseline, {metric}_delta for each metric
        timepoints: Days to predict
        metric_names: List of metric names (if None, auto-detect from columns)
    
    Returns:
        DataFrame with columns:
            - patient_id, drug_name (derived from drug_SMILES or original drug_name), metric_name
            - day_10, day_20, day_30, day_60, day_90, day_180 (predicted deltas)
    """
    if metric_names is None:
        # Auto-detect metric names from columns ending in '_delta'
        metric_names = [col.replace('_delta', '') for col in predictor_data.columns 
                       if col.endswith('_delta') and not col.startswith('day_')]
    
    # Metric type mapping
    metric_types = {
        'LBXTC': 'lipid', 'LBDLDL': 'lipid', 'LBXSCH': 'lipid',
        'LBXSGL': 'glucose', 'LBXGH': 'glucose',
        'BPXSY1': 'bp', 'BPXDI1': 'bp',
        'LBDSCR': 'renal', 'LBXSBU': 'renal',
        'LBXSK': 'electrolyte', 'LBXSNA': 'electrolyte', 'LBXSUA': 'electrolyte',
        'LBXSAS': 'liver', 'LBXSAL': 'liver', 'LBXSGB': 'liver',
    }
    
    results = []
    
    for _, row in predictor_data.iterrows():
        patient_id = row.get('patient_id', 'unknown')
        # Support both drug_SMILES and drug_name for backward compatibility
        if 'drug_SMILES' in row:
            drug_smiles = row['drug_SMILES']
            # Use SMILES as drug identifier (use default drug class for trajectory)
            drug_name = drug_smiles[:20] if len(str(drug_smiles)) > 20 else str(drug_smiles)
        else:
            drug_name = row.get('drug_name', 'Unknown')
        adherence = row.get('adherence', 1.0)
        days_on_drug = row.get('days_on_drug', 180)
        
        for metric_base in metric_names:
            baseline_col = f'{metric_base}_baseline'
            delta_col = f'{metric_base}_delta'
            
            # Skip if missing
            if baseline_col not in row or delta_col not in row:
                continue
            
            final_delta = row[delta_col]
            
            # Skip if delta is NaN or zero
            if pd.isna(final_delta) or abs(final_delta) < 0.01:
                continue
            
            # Get metric type
            metric_type = metric_types.get(metric_base, 'default')
            
            # Generate trajectory (uses 'Default' drug class since we don't have drug names)
            trajectory = generate_trajectory(
                final_delta=final_delta,
                timepoints=timepoints,
                drug_name='Default',  # Use default profile since we only have SMILES
                adherence=adherence,
                days_on_drug=days_on_drug,
                metric_type=metric_type,
                use_sigmoid=True,
                noise_level=0.05,
                patient_variability=0.15
            )
            
            # Create result row
            result = {
                'patient_id': patient_id,
                'drug_name': drug_name,  # Store SMILES or drug name
                'metric_name': metric_base,
                'final_delta': final_delta,
                'baseline': row[baseline_col],
            }
            
            # Add trajectory values
            for day, delta in zip(timepoints, trajectory):
                result[f'day_{day}'] = delta
            
            results.append(result)
    
    return pd.DataFrame(results)


def validate_trajectory(
    trajectory: np.ndarray,
    final_delta: float,
    timepoints: List[int],
    tolerance: float = 0.1
) -> bool:
    """
    Validate that trajectory is pharmacologically plausible.
    
    Checks:
    1. Monotonicity (for most metrics, change should be monotonic)
    2. Final value should be close to final_delta
    3. No sudden jumps
    """
    # Check final value
    final_predicted = trajectory[-1]
    if abs(final_predicted - final_delta) > abs(final_delta) * tolerance:
        return False
    
    # Check monotonicity (for decreasing metrics)
    if final_delta < 0:
        if not np.all(np.diff(trajectory) <= 0.1):  # Allow small noise
            return False
    else:
        if not np.all(np.diff(trajectory) >= -0.1):
            return False
    
    # Check for sudden jumps (change > 50% of final delta in one step)
    diffs = np.abs(np.diff(trajectory))
    max_jump = abs(final_delta) * 0.5
    if np.any(diffs > max_jump):
        return False
    
    return True

