"""Temporal dynamics simulator that consumes S0, drug embedding and predictor.

Simulates repeated application of the pharmacodynamic predictor at fixed
time intervals with feature-specific time constants and stochastic noise
for realistic dynamics. Returns a time-series DataFrame of states.

Updated to handle partial state updates:
- Predictor returns changes to 22 lab biomarkers only
- Simulator mutates ONLY lab features (indices 12-33)
- Body measurements and questionnaires remain frozen

NEW: Time Series Prediction from Static Drug Response Data
- Converts static predictions (baseline + final_delta) into temporal trajectories
- Uses pharmacologically realistic sigmoid/exponential curves
- Supports drug-specific response profiles and adherence modeling
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.lab_reference_ranges import (
    LAB_RANGES, CRITICAL_RULES, get_lab_range, 
    validate_lab_value, clamp_to_physiological_limits, get_sex_adjusted_range
)

# Lab biomarker feature names (22 features that get updated)
LAB_BIOMARKER_FEATURES = [
    'LBDSCASI', 'LBDSCH', 'LBDSCR', 'LBDTC',
    'LBXBCD', 'LBXBPB', 'LBXCRP', 'LBXSAL',
    'LBXSAS', 'LBXSBU', 'LBXSCA', 'LBXSCH',
    'LBXSCL', 'LBXSGB', 'LBXSGL', 'LBXSGT',
    'LBXSK', 'LBXSNA', 'LBXSOS', 'LBXSTP',
    'LBXSUA', 'LBXTC'
]

# Drug-specific response profiles for temporal modeling
DRUG_RESPONSE_PROFILES = {
    'Statin': {'tau_days': 30, 'steepness': 2.0, 'early_lag_days': 3},
    'ACE_Inhibitor': {'tau_days': 14, 'steepness': 3.0, 'early_lag_days': 1},
    'Metformin': {'tau_days': 60, 'steepness': 1.5, 'early_lag_days': 7},
    'Diuretic': {'tau_days': 7, 'steepness': 4.0, 'early_lag_days': 0.5},
    'Antibiotic': {'tau_days': 1, 'steepness': 5.0, 'early_lag_days': 0},
    'Default': {'tau_days': 21, 'steepness': 2.5, 'early_lag_days': 2}
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

# Metric type mapping for response speed adjustments
METRIC_TYPES = {
    'LBXTC': 'lipid', 'LBDLDL': 'lipid', 'LBXSCH': 'lipid',
    'LBXSGL': 'glucose', 'LBXGH': 'glucose',
    'BPXSY1': 'bp', 'BPXDI1': 'bp',
    'LBDSCR': 'renal', 'LBXSBU': 'renal',
    'LBXSK': 'electrolyte', 'LBXSNA': 'electrolyte',
}


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


def generate_trajectory_from_static(
    final_delta: float,
    timepoints: List[int],
    drug_name: str,
    adherence: float = 1.0,
    metric_name: Optional[str] = None,
    baseline: Optional[float] = None,
    use_sigmoid: bool = True,
    noise_level: float = 0.05,
    patient_variability: float = 0.2
) -> np.ndarray:
    """
    Generate realistic trajectory for a single metric from static prediction.
    
    Converts static prediction (baseline + final_delta) into temporal trajectory
    using pharmacologically realistic curves.
    
    Args:
        final_delta: Final delta value from static predictor (negative = decrease, positive = increase)
        timepoints: List of days to predict (e.g., [10, 20, 30, 60, 90, 180])
        drug_name: Name of drug
        adherence: Medication adherence (0-1)
        metric_name: Name of metric (e.g., 'LBXTC', 'LBXSGL') for type-specific adjustments
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
    if metric_name:
        metric_type = METRIC_TYPES.get(metric_name, 'default')
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
        # Exponential alternative
        trajectory = final_delta * (1.0 - np.exp(-t / tau))
    
    # Apply adherence modulation
    # Non-adherent patients have slower/less complete responses
    trajectory = trajectory * adherence
    
    # Add extra variability for non-adherent patients
    if adherence < 0.9:
        variability = (1.0 - adherence) * 0.1
        noise = np.random.normal(0, variability * np.abs(final_delta), size=trajectory.shape)
        trajectory += noise
    
    # Add final noise
    if noise_level > 0:
        noise = np.random.normal(0, abs(final_delta) * noise_level, size=trajectory.shape)
        trajectory += noise
    
    # Ensure trajectory doesn't exceed final_delta (for decreasing metrics)
    if final_delta < 0:
        trajectory = np.maximum(trajectory, final_delta)
    else:
        trajectory = np.minimum(trajectory, final_delta)
    
    # Apply physiological limits to ensure absolute values stay within realistic bounds
    if metric_name and metric_name in LAB_RANGES and baseline is not None:
        phys_limit = LAB_RANGES[metric_name]['physiological_limit']
        if phys_limit:
            min_phys, max_phys = phys_limit
            for i in range(len(trajectory)):
                absolute_value = baseline + trajectory[i]
                # Clamp absolute value to physiological limits
                if absolute_value < min_phys:
                    trajectory[i] = min_phys - baseline
                elif absolute_value > max_phys:
                    trajectory[i] = max_phys - baseline
    
    return trajectory


def predict_time_series(
    predictor_data: pd.DataFrame,
    timepoints: List[int] = [10, 20, 30, 60, 90, 180],
    metric_names: Optional[List[str]] = None,
    return_absolute: bool = False
) -> pd.DataFrame:
    """
    Convert static drug response predictions into temporal trajectories.
    
    Takes DataFrame with baseline + final_delta columns and generates
    time series showing how delta evolves over time.
    
    Args:
        predictor_data: DataFrame with columns:
            - patient_id, drug_name, age, sex, bmi, days_on_drug, adherence
            - {metric}_baseline, {metric}_delta for each metric
        timepoints: Days to predict (default: [10, 20, 30, 60, 90, 180])
        metric_names: List of metric names (if None, auto-detect from columns)
        return_absolute: If True, return absolute values (baseline + delta)
                        If False, return delta values only
    
    Returns:
        DataFrame with columns:
            - patient_id, drug_name, metric_name
            - day_10, day_20, day_30, day_60, day_90, day_180 (predicted deltas or absolute values)
    """
    if metric_names is None:
        # Auto-detect metric names from columns ending in '_delta'
        metric_names = [col.replace('_delta', '') for col in predictor_data.columns 
                       if col.endswith('_delta') and not col.startswith('day_')]
    
    results = []
    
    for _, row in predictor_data.iterrows():
        patient_id = row.get('patient_id', 'unknown')
        drug_name = row.get('drug_name', 'Unknown')
        adherence = row.get('adherence', 1.0)
        days_on_drug = row.get('days_on_drug', 180)
        
        for metric_base in metric_names:
            baseline_col = f'{metric_base}_baseline'
            delta_col = f'{metric_base}_delta'
            
            # Skip if missing
            if baseline_col not in row or delta_col not in row:
                continue
            
            baseline = row[baseline_col]
            final_delta = row[delta_col]
            
            # Skip if delta is NaN or zero
            if pd.isna(final_delta) or abs(final_delta) < 0.01:
                continue
            
            # Validate baseline is within physiological limits
            if metric_base in LAB_RANGES:
                baseline = clamp_to_physiological_limits(metric_base, baseline)
                # Ensure final value (baseline + delta) is also within limits
                final_value = baseline + final_delta
                final_value_clamped = clamp_to_physiological_limits(metric_base, final_value)
                # Adjust delta if final value was clamped
                final_delta = final_value_clamped - baseline
            
            # Generate trajectory with baseline for physiological validation
            trajectory = generate_trajectory_from_static(
                final_delta=final_delta,
                timepoints=timepoints,
                drug_name=drug_name,
                adherence=adherence,
                metric_name=metric_base,
                baseline=baseline,  # Pass baseline for absolute value validation
                use_sigmoid=True,
                noise_level=0.05,
                patient_variability=0.15
            )
            
            # Create result row
            result = {
                'patient_id': patient_id,
                'drug_name': drug_name,
                'metric_name': metric_base,
                'baseline': baseline,
                'final_delta': final_delta,
            }
            
            # Add trajectory values
            for day, delta in zip(timepoints, trajectory):
                if return_absolute:
                    result[f'day_{day}'] = baseline + delta
                else:
                    result[f'day_{day}'] = delta
            
            results.append(result)
    
    return pd.DataFrame(results)


def simulate_from_clinical_trials_data(
    clinical_data: pd.DataFrame,
    timepoints: List[int] = [10, 20, 30, 60, 90, 180],
    metric_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    High-level function to generate time series from clinical trials data format.
    
    This is the main entry point for converting static predictions into temporal trajectories.
    
    Args:
        clinical_data: DataFrame with format from clinical_trial_50k.csv:
            - patient_id, drug_SMILES (or drug_name), age, sex, bmi, days_on_drug, adherence
            - {metric}_baseline, {metric}_delta for each metric
        timepoints: Days to predict
        metric_names: List of metric names (if None, auto-detect)
    
    Returns:
        DataFrame with time series predictions for each patient-metric combination
    
    Example:
        >>> import pandas as pd
        >>> data = pd.read_csv('data/cdisc/clinical_trial_50k.csv')
        >>> trajectories = simulate_from_clinical_trials_data(data)
        >>> # trajectories has columns: patient_id, drug_name, metric_name, day_10, day_20, ...
    """
    return predict_time_series(clinical_data, timepoints, metric_names, return_absolute=False)


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import pandas as pd
    
    # Example 1: Original simulate function (minutes-based)
    print("=" * 60)
    print("Example 1: Original simulate function (minutes-based)")
    print("=" * 60)
    try:
        from models.pharmacodynamicPredictor import PharmacodynamicPredictor
        fn = [f'F{i}' for i in range(5)]
        s0 = pd.DataFrame([[1,2,3,4,5]], columns=fn)
        pred = PharmacodynamicPredictor(fn, embed_dim=32)
        emb = np.ones(32)
        result = simulate(s0, emb, pred, steps=3, dt_min=5)
        print(result)
    except Exception as e:
        print(f"Original simulate example failed: {e}")
    
    # Example 2: Time series prediction from static predictions (days-based)
    print("\n" + "=" * 60)
    print("Example 2: Time series prediction from static predictions")
    print("=" * 60)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'patient_id': [1, 2, 3],
        'drug_name': ['Simvastatin', 'Metformin', 'Lisinopril'],
        'age': [55, 60, 65],
        'sex': ['M', 'F', 'M'],
        'bmi': [28.0, 32.0, 26.0],
        'days_on_drug': [90, 120, 60],
        'adherence': [0.95, 0.85, 0.90],
        'LBXTC_baseline': [220.0, 200.0, 210.0],
        'LBXTC_delta': [-35.0, -5.0, 0.0],
        'LBXSGL_baseline': [110.0, 150.0, 105.0],
        'LBXSGL_delta': [0.0, -25.0, 0.0],
        'BPXSY1_baseline': [140.0, 135.0, 145.0],
        'BPXSY1_delta': [0.0, 0.0, -12.0],
    })
    
    print("\nInput data (static predictions):")
    print(sample_data[['patient_id', 'drug_name', 'LBXTC_baseline', 'LBXTC_delta', 
                       'LBXSGL_baseline', 'LBXSGL_delta']])
    
    # Generate time series
    trajectories = predict_time_series(
        sample_data,
        timepoints=[10, 20, 30, 60, 90, 180],
        metric_names=['LBXTC', 'LBXSGL', 'BPXSY1']
    )
    
    print("\nOutput: Time series trajectories")
    print(trajectories[['patient_id', 'drug_name', 'metric_name', 'day_10', 'day_30', 'day_90', 'day_180']].head(10))
    
    # Example 3: Single trajectory visualization
    print("\n" + "=" * 60)
    print("Example 3: Single trajectory generation")
    print("=" * 60)
    
    # Patient on Simvastatin: cholesterol decreases from 220 to 185 (delta = -35)
    trajectory = generate_trajectory_from_static(
        final_delta=-35.0,
        timepoints=[10, 20, 30, 60, 90, 180],
        drug_name='Simvastatin',
        adherence=0.95,
        metric_name='LBXTC',
        noise_level=0.05
    )
    
    print(f"\nCholesterol trajectory (baseline=220, final_delta=-35):")
    for day, delta in zip([10, 20, 30, 60, 90, 180], trajectory):
        absolute = 220 + delta
        print(f"  Day {day:3d}: delta={delta:6.2f}, absolute={absolute:6.2f} mg/dL")
    
    print("\n" + "=" * 60)
    print("Time series prediction complete!")
    print("=" * 60)
