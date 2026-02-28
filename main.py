"""Orchestration script: End-to-end drug simulation pipeline with time series prediction.

Usage: 
    python main.py --smiles <SMILES> --age <AGE> --sex <M/F> --height <CM> --weight <KG> [--adherence <0-1>] [--days <DAYS>]
    
    Or set parameters in the script directly (see PATIENT_CONFIG and DRUG_CONFIG below)

Pipeline:
1. Generate synthetic patient using GAN (best model)
2. Encode drug SMILES
3. Predict pharmacodynamic response (static: baseline + final_delta)
4. Convert static predictions to time series trajectories
5. Save results to ./results/time_series_simulation.csv

NEW: Time Series Prediction
- Converts static predictions into temporal trajectories
- Shows how drug response evolves over days (10, 20, 30, 60, 90, 180)
- Uses pharmacologically realistic sigmoid curves
- Accounts for drug-specific response profiles and adherence
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import List
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.patient_generator_gan import PatientGenerator
from encoders.drugEncoder import DrugEncoder
from models.pharmacodynamicPredictor import PharmacodynamicPredictor, LAB_BIOMARKER_FEATURES
from models.time_series_predictor import TimeSeriesPredictor, TimeSeriesConfig
from pathlib import Path

# Configuration
OUTPUT_DIR = Path('results')
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

# ============================================================================
# PATIENT CONFIGURATION - Will be prompted for input
# ============================================================================
PATIENT_CONFIG = {}
DRUG_CONFIG = {}

# Time points for time series prediction (seconds)
# Single injection scenario: 0 to 3 hours (10800 seconds)
SECONDS_PER_HOUR = 3600
TOTAL_HOURS = 3
TOTAL_SECONDS = TOTAL_HOURS * SECONDS_PER_HOUR  # 10800 seconds (3 hours)

def generate_timepoints_seconds(total_seconds: int = TOTAL_SECONDS, interval_seconds: int = 10) -> List[int]:
    """
    Generate timepoints every N seconds up to total_seconds (for single injection).
    
    Args:
        total_seconds: Maximum time in seconds (default: 10800 = 3 hours)
        interval_seconds: Interval between timepoints in seconds (default: 10 seconds)
    
    Returns:
        List of timepoints in seconds, e.g., [10, 20, 30, ..., 10800]
    """
    # Always include 0 (baseline)
    timepoints = [0]
    
    # Add timepoints every 'interval_seconds' seconds, up to total_seconds
    current_second = interval_seconds
    while current_second <= total_seconds:
        timepoints.append(current_second)
        current_second += interval_seconds
    
    # Ensure the final second is included (might not be exactly divisible by interval)
    if timepoints[-1] != total_seconds:
        timepoints.append(total_seconds)
    
    # Remove duplicates and sort
    timepoints = sorted(list(set(timepoints)))
    
    return timepoints

# Default timepoints (for backward compatibility)
TIMEPOINTS = [10, 20, 30, 60, 90, 180]

# Time format: "X Y" where X is a number, Y is unit: sec, min, h, day
TIME_FORMAT_HELP = "Format: X Y  (X=number, Y=sec|min|h|day).  Examples: 3 h, 90 min, 10 sec"

def parse_time_to_seconds(s: str) -> float:
    """
    Parse time string "X Y" to seconds.
    X: number (int or float).  Y: sec, min, h, day (case-insensitive).
    Examples: "3 h" -> 10800, "10 sec" -> 10, "1.5 min" -> 90.
    """
    s = s.strip().lower()
    parts = s.split()
    if len(parts) < 2:
        raise ValueError(f"Expected 'X Y' (e.g. 3 h, 10 sec). Got: {s!r}")
    try:
        x = float(parts[0])
    except ValueError:
        raise ValueError(f"X must be a number, got: {parts[0]!r}")
    unit = parts[1]
    if unit in ("sec", "secs", "second", "seconds"):
        return x
    if unit in ("min", "mins", "minute", "minutes"):
        return x * 60
    if unit in ("h", "hour", "hours", "hr", "hrs"):
        return x * 3600
    if unit in ("d", "day", "days"):
        return x * 86400
    raise ValueError(f"Unit must be sec, min, h, or day. Got: {unit!r}")


def get_user_inputs():
    """Prompt user for all patient and drug parameters."""
    print("\n" + "="*70)
    print("PATIENT AND DRUG CONFIGURATION")
    print("="*70)
    
    # Patient demographics
    print("\n--- PATIENT DEMOGRAPHICS ---")
    while True:
        try:
            age = int(input("Patient age (years, 18-85): "))
            if 18 <= age <= 85:
                break
            else:
                print("  ERROR: Age must be between 18 and 85")
        except ValueError:
            print("  ERROR: Please enter a valid number")
    
    while True:
        sex = input("Patient sex (M/F): ").strip().upper()
        if sex in ['M', 'F']:
            break
        else:
            print("  ERROR: Must be 'M' or 'F'")
    
    while True:
        try:
            height = float(input("Patient height (cm, 130-200): "))
            if 130 <= height <= 200:
                break
            else:
                print("  ERROR: Height must be between 130 and 200 cm")
        except ValueError:
            print("  ERROR: Please enter a valid number")
    
    while True:
        try:
            weight = float(input("Patient weight (kg, 30-250): "))
            if 30 <= weight <= 250:
                break
            else:
                print("  ERROR: Weight must be between 30 and 250 kg")
        except ValueError:
            print("  ERROR: Please enter a valid number")
    
    # Treatment parameters
    print("\n--- TREATMENT PARAMETERS (Single Injection) ---")
    while True:
        try:
            dosage = float(input("Drug dosage per injection (mg, e.g., 10.0, 20.0, 50.0): "))
            if dosage > 0:
                break
            else:
                print("  ERROR: Dosage must be greater than 0")
        except ValueError:
            print("  ERROR: Please enter a valid number")
    
    # Simulation duration and checkpoint interval (format: X Y — X number, Y sec|min|h|day)
    print("\n--- TIME (format: X Y  e.g. 3 h, 10 sec) ---")
    default_duration = "3 h"
    while True:
        raw = input(f"Simulation duration / final time (default {default_duration}): ").strip()
        if not raw:
            raw = default_duration
        try:
            total_seconds_d = parse_time_to_seconds(raw)
            total_hours = total_seconds_d / 3600.0
            if 0.1 <= total_hours <= 24.0:
                break
            print("  ERROR: Duration must be between 0.1 and 24 hours (after converting to hours)")
        except ValueError as e:
            print(f"  ERROR: {e}")
            print(f"  {TIME_FORMAT_HELP}")

    default_interval = "10 sec"
    while True:
        raw = input(f"Checkpoint every (default {default_interval}): ").strip()
        if not raw:
            raw = default_interval
        try:
            interval_seconds = int(round(parse_time_to_seconds(raw)))
            if interval_seconds < 1:
                print("  ERROR: Interval must be at least 1 second")
                continue
            if interval_seconds > total_seconds_d:
                print("  ERROR: Interval must not exceed simulation duration")
                continue
            break
        except ValueError as e:
            print(f"  ERROR: {e}")
            print(f"  {TIME_FORMAT_HELP}")

    # Set default adherence (not used for single injection but kept for model compatibility)
    adherence = 1.0
    
    # Drug information
    print("\n--- DRUG INFORMATION ---")
    smiles = input("Drug SMILES string: ").strip()
    if not smiles:
        print("  ERROR: SMILES string cannot be empty")
        sys.exit(1)
    
    drug_name = input("Drug name (optional, press Enter for 'Unknown'): ").strip()
    if not drug_name:
        drug_name = 'Unknown'
    
    print("="*70 + "\n")
    
    return {
        'patient': {
            'age': age,
            'sex': sex,
            'height': height,
            'weight': weight,
            'adherence': adherence,  # Not used for single injection, but kept for compatibility
            'dosage': dosage
        },
        'drug': {
            'smiles': smiles,
            'drug_name': drug_name
        },
        'total_hours': total_hours,
        'interval_seconds': interval_seconds,
    }


def parse_arguments():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Drug simulation with time series prediction')
    parser.add_argument('--smiles', type=str, help='Drug SMILES string')
    parser.add_argument('--drug_name', type=str, help='Drug name (optional, for better response profiles)')
    parser.add_argument('--age', type=int, help='Patient age (years)')
    parser.add_argument('--sex', type=str, choices=['M', 'F'], help='Patient sex')
    parser.add_argument('--height', type=float, help='Patient height (cm)')
    parser.add_argument('--weight', type=float, help='Patient weight (kg)')
    parser.add_argument('--adherence', type=float, help='Medication adherence (0-1)')
    parser.add_argument('--days', type=int, help='Days on drug')
    parser.add_argument('--dosage', type=float, help='Drug dosage in mg')
    parser.add_argument('--total_time', type=str, default='3 h', help='Simulation duration: X Y (X=number, Y=sec|min|h|day). Default: 3 h')
    parser.add_argument('--interval', type=str, default='10 sec', help='Checkpoint every: X Y (X=number, Y=sec|min|h|day). Default: 10 sec')
    parser.add_argument('--no-input', action='store_true', help='Skip interactive input (use defaults)')
    return parser.parse_args()


def check_and_train_models():
    """Check for required models and train ONLY the missing ones."""
    import subprocess
    import sys
    
    def check_gan_model_exists():
        """Check if GAN model exists in either location."""
        path1 = Path('models/patient_generator_gan_best.pt')
        path2 = Path('models/generator/patient_generator_gan_best.pt')
        return path1.exists() or path2.exists()
    
    models_to_check = [
        {
            'path': Path('models/generator/patient_generator_gan_best.pt'),  # Actual save location
            'alt_path': Path('models/patient_generator_gan_best.pt'),  # Alternative location
            'check_func': check_gan_model_exists,  # Custom check function
            'training_script': 'src/training/train_gan.py',
            'name': 'Patient Generator GAN',
            'required': False,  # Optional, has fallback
            'order': 1  # Train first (independent)
        },
        {
            'path': Path('models/pharmacodynamic_predictor/predictor_novel_drug_best.pt'),
            'alt_path': None,
            'check_func': None,
            'training_script': 'src/training/train_predictor.py',
            'name': 'Pharmacodynamic Predictor',
            'required': True,
            'order': 2  # Train second (independent)
        },
        {
            'path': Path('models/time_series_predictor.pt'),
            'alt_path': None,
            'check_func': None,
            'training_script': 'src/training/train_time_series_predictor.py',
            'name': 'Time Series Predictor',
            'required': True,
            'order': 3  # Train last
        }
    ]
    
    # Check which models are missing
    missing_models = []
    for model in models_to_check:
        if model['check_func']:
            # Use custom check function
            exists = model['check_func']()
        else:
            # Standard path check
            exists = model['path'].exists()
            if not exists and model['alt_path']:
                exists = model['alt_path'].exists()
        
        if not exists:
            missing_models.append(model)
    
    if not missing_models:
        print("\n" + "="*70)
        print("ALL MODELS EXIST")
        print("="*70)
        print("\n✓ All required models are present. No training needed.")
        print("="*70 + "\n")
        return  # All models exist
    
    # Sort missing models by training order
    missing_models.sort(key=lambda x: x['order'])
    
    print("\n" + "="*70)
    print("MISSING MODELS DETECTED")
    print("="*70)
    print(f"\nFound {len(missing_models)} missing model(s):")
    for model in missing_models:
        status = "REQUIRED" if model['required'] else "Optional"
        print(f"  - {model['name']} ({status})")
        print(f"    Path: {model['path']}")
        print(f"    Training script: {model['training_script']}")
    
    print("\n" + "="*70)
    print("TRAINING ONLY MISSING MODELS")
    print("="*70)
    print(f"\nTraining {len(missing_models)} missing model(s) now. This may take a while...")
    print("="*70 + "\n")
    
    # Train ONLY the missing models in order
    for model_info in missing_models:
        # Double-check if model exists (might have been created by a previous training step)
        if model_info['check_func']:
            exists = model_info['check_func']()
        else:
            exists = model_info['path'].exists()
            if not exists and model_info['alt_path']:
                exists = model_info['alt_path'].exists()
        
        if exists:
            print(f"\n✓ {model_info['name']} already exists")
            print(f"  Skipping training for this model.")
            continue
        
        print(f"\n{'='*70}")
        print(f"Training: {model_info['name']}")
        print(f"Script: {model_info['training_script']}")
        print(f"{'='*70}\n")
        
        try:
            # Run training script
            result = subprocess.run(
                [sys.executable, model_info['training_script']],
                check=True,
                cwd=Path(__file__).parent
            )
            print(f"\n✓ Successfully trained {model_info['name']}")
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to train {model_info['name']}"
            if model_info['required']:
                raise RuntimeError(
                    f"\n{'='*70}\n"
                    f"ERROR: {error_msg}\n"
                    f"{'='*70}\n"
                    f"Training script: {model_info['training_script']}\n"
                    f"Exit code: {e.returncode}\n\n"
                    f"This model is REQUIRED. Please check the training script for errors.\n"
                    f"{'='*70}"
                ) from e
            else:
                print(f"\n⚠ Warning: {error_msg} (optional model, continuing...)")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"\n{'='*70}\n"
                f"ERROR: Training script not found: {model_info['training_script']}\n"
                f"{'='*70}\n"
                f"Please ensure the training script exists.\n"
                f"{'='*70}"
            )
    
    # Verify all required models were trained
    print("\n" + "="*70)
    print("VERIFYING TRAINED MODELS")
    print("="*70)
    all_trained = True
    for model in models_to_check:
        if model['required']:
            if model['check_func']:
                exists = model['check_func']()
            else:
                exists = model['path'].exists()
                if not exists and model['alt_path']:
                    exists = model['alt_path'].exists()
            
            if exists:
                print(f"  ✓ {model['name']}: {model['path']}")
            else:
                print(f"  ✗ {model['name']}: MISSING - {model['path']}")
                all_trained = False
    
    if not all_trained:
        raise RuntimeError(
            f"\n{'='*70}\n"
            f"ERROR: Some required models failed to train or are still missing!\n"
            f"{'='*70}\n"
            f"Please check the training logs above for errors.\n"
            f"{'='*70}"
        )
    
    print("\n✓ All required models are ready!")
    print("="*70 + "\n")


def run(smiles: str, drug_name: str = 'Unknown', age: int = 55, sex: str = 'M', 
        height: float = 175.0, weight: float = 80.0, adherence: float = 0.95,
        dosage: float = 10.0, timepoints_seconds: List[int] = None, 
        interval_seconds: int = 10, total_hours: float = 3.0):
    """
    Run full simulation pipeline with time series prediction for single injection.
    
    Args:
        smiles: Drug SMILES string
        drug_name: Drug name (optional, for labeling)
        age: Patient age in years
        sex: Patient sex ('M' or 'F')
        height: Height in cm
        weight: Weight in kg
        adherence: Medication adherence (0-1, not used for single injection but kept for compatibility)
        dosage: Drug dosage per injection in mg
        timepoints_seconds: Seconds to predict (if None, will generate every 'interval_seconds' up to total_hours * 3600)
        interval_seconds: Seconds between timepoints (default: 10 seconds)
        total_hours: Total time to simulate in hours (default: 3 hours)
    
    Returns:
        DataFrame: Time series trajectories with columns: patient_id, metric_name, baseline, final_delta, 10sec, 20sec, ...
    """
    # Calculate total seconds (3 hours by default)
    total_seconds = int(total_hours * SECONDS_PER_HOUR)  # 10800 seconds = 3 hours
    
    # Generate timepoints in seconds dynamically if not provided
    if timepoints_seconds is None:
        timepoints_seconds = generate_timepoints_seconds(total_seconds, interval_seconds=interval_seconds)
        # Remove 0 from timepoints (baseline is separate)
        timepoints_seconds = [t for t in timepoints_seconds if t > 0]
    
    # Convert seconds to days for model prediction (model expects days)
    # We'll use the model's day-based predictions and interpolate to seconds
    # For single injection over 3 hours, we'll scale the model's day-based predictions
    model_timepoints_days = [10, 20, 30, 60, 90, 180]  # Model's trained timepoints in days
    
    # Check and train missing models before starting
    check_and_train_models()
    
    print("\n" + "="*70)
    print("DRUG SIMULATION PIPELINE WITH TIME SERIES PREDICTION")
    print("="*70)
    
    # Step 1: Generate synthetic patient using best GAN model
    print("\n[1/4] Generating synthetic patient...")
    gen = PatientGenerator(data_path='preprocessed_nhanes/nhanes_final_complete.csv')
    
    # Load best model (trained with FID metric) - check both possible locations
    best_model_path = Path('models/generator/patient_generator_gan_best.pt')
    if not best_model_path.exists():
        best_model_path = Path('models/patient_generator_gan_best.pt')
    
    if best_model_path.exists():
        import torch
        checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
        gen.generator.load_state_dict(checkpoint['generator'])
        print(f"  ✓ Loaded best model from epoch {checkpoint.get('epoch', '?')}")
        print(f"    FID score: {checkpoint.get('fid_score', '?'):.2f}")
    else:
        print("  ! Using default model (best model not found)")
    
    # Generate patient with specified demographics
    S0 = gen.generate(age=age, sex=sex, height=height, weight=weight, n=1, seed=42)
    print(f"  ✓ Generated patient: {age}yo {sex}, {height}cm, {weight}kg")
    print(f"    BMI: {S0['BMXBMI'].values[0]:.1f}")
    print(f"    Cholesterol: {S0['LBXTC'].values[0]:.1f} mg/dL")
    print(f"    Glucose: {S0['LBXSGL'].values[0]:.1f} mg/dL")
    
    # Step 2: Encode drug
    print("\n[2/4] Encoding drug...")
    try:
        enc = DrugEncoder(encoder_type='hybrid', device=DEVICE)
        emb = enc.encode(smiles)
        print(f"  ✓ Drug encoded: {smiles}")
        print(f"    Embedding dimension: {emb.shape[0]}")
        print(f"    Embedding norm: {np.linalg.norm(emb):.2f}")
        print(f"    Encoder: Hybrid (neural + RDKit descriptors)")
    except Exception as e:
        print(f"  ✗ Drug encoder failed: {e}")
        raise
    
    # Step 3: Predict pharmacodynamic response (static prediction)
    print("\n[3/4] Predicting pharmacodynamic response (static)...")
    predictor = PharmacodynamicPredictor(
        predictor_type='cross_attention',
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        use_constraints=True
    )
    
    # Try to load trained weights if available
    predictor_model_path = Path('models/pharmacodynamic_predictor/predictor_novel_drug_best.pt')
    if predictor_model_path.exists():
        try:
            predictor.load(str(predictor_model_path))
            print(f"  ✓ Loaded trained Pharmacodynamic Predictor from {predictor_model_path}")
        except Exception as e:
            print(f"  ⚠ Warning: Failed to load trained predictor weights: {e}")
            print(f"    Using untrained model (predictions may be less accurate)")
    else:
        print(f"  ⚠ Warning: Trained predictor weights not found at {predictor_model_path}")
        print(f"    Using untrained model (predictions may be less accurate)")
    
    print(f"  ✓ Cross-attention predictor initialized")
    
    # Get static predictions (baseline + final_delta)
    delta_df = predictor.predict_delta(S0, emb)
    print(f"  ✓ Predicted changes for {len(LAB_BIOMARKER_FEATURES)} lab biomarkers")
    
    # Extract baseline values and deltas
    predictor_data = {
        'patient_id': [1],
        'drug_name': [drug_name],
        'age': [age],
        'sex': [sex],
        'bmi': [S0['BMXBMI'].values[0]],
        'days_on_drug': [1],  # Dummy value for model compatibility (not used for single injection)
        'adherence': [adherence],
        'dosage': [dosage],  # Add dosage
    }
    
    # Add baseline and delta for each metric
    for metric in LAB_BIOMARKER_FEATURES:
        if metric in S0.columns:
            baseline = S0[metric].values[0]
            delta = delta_df[metric].values[0] if metric in delta_df.columns else 0.0
            predictor_data[f'{metric}_baseline'] = [baseline]
            predictor_data[f'{metric}_delta'] = [delta]
    
    predictor_df = pd.DataFrame(predictor_data)
    
    # Show key predictions
    print("\n  Key Predictions:")
    key_metrics = ['LBXTC', 'LBXSGL', 'LBDSCR', 'BPXSY1', 'BPXDI1']
    for metric in key_metrics:
        if f'{metric}_baseline' in predictor_df.columns:
            baseline = predictor_df[f'{metric}_baseline'].values[0]
            delta = predictor_df[f'{metric}_delta'].values[0]
            final = baseline + delta
            print(f"    {metric}: {baseline:.1f} → {final:.1f} (Δ={delta:+.1f})")
    
    # Step 4: Convert static predictions to time series using trained model (MANDATORY)
    print("\n[4/4] Converting static predictions to time series trajectories...")
    
    # Load trained Time Series Predictor model (should exist after check_and_train_models)
    time_series_model_path = Path('models/time_series_predictor.pt')
    if not time_series_model_path.exists():
        # This should not happen if check_and_train_models worked correctly
        raise FileNotFoundError(
            f"\n{'='*70}\n"
            f"ERROR: Time Series Predictor model not found!\n"
            f"{'='*70}\n"
            f"Required model: {time_series_model_path}\n\n"
            f"This model is MANDATORY for converting static predictions to time series.\n"
            f"Automatic training should have created this file.\n"
            f"If this error persists, please train manually:\n"
            f"  python src/training/train_time_series_predictor.py\n\n"
            f"The Time Series Predictor converts static predictions (from PharmacodynamicPredictor)\n"
            f"into temporal trajectories - it does NOT predict drug effects itself.\n"
            f"{'='*70}"
        )
    
    # Initialize Time Series Predictor with dummy parameters (will be overwritten by load)
    # Load will recreate the model with correct dimensions from checkpoint
    # Model uses day-based timepoints, we'll convert to seconds after prediction
    dummy_drug_vocab = {'Unknown': 0}
    dummy_metric_names = ['LBXTC']  # Dummy, will be replaced by checkpoint
    time_series_predictor = TimeSeriesPredictor(
        drug_vocab=dummy_drug_vocab,
        metric_names=dummy_metric_names,
        patient_feature_names=['age', 'sex', 'bmi', 'adherence', 'days_on_drug', 'dosage'],
        config=TimeSeriesConfig(timepoints=model_timepoints_days),  # Use model's day-based timepoints
        device=DEVICE
    )
    
    # Load trained weights (this will recreate model with checkpoint dimensions)
    try:
        time_series_predictor.load(str(time_series_model_path))
        print(f"  ✓ Loaded trained Time Series Predictor from {time_series_model_path}")
        print(f"    Model metrics: {', '.join(time_series_predictor.metric_names)}")
        print(f"    Model drugs: {len(time_series_predictor.drug_vocab)} drugs in vocabulary")
    except Exception as e:
        raise RuntimeError(
            f"\n{'='*70}\n"
            f"ERROR: Failed to load Time Series Predictor model!\n"
            f"{'='*70}\n"
            f"Model path: {time_series_model_path}\n"
            f"Error: {e}\n\n"
            f"Please ensure the model is properly trained:\n"
            f"  python src/training/train_time_series_predictor.py\n"
            f"{'='*70}"
        )
    
    # Use SMILES-based drug embedding directly instead of drug name vocabulary
    # The drug embedding from DrugEncoder (768 dim) will be used
    # We'll pass it to the predictor and it will handle the projection
    print(f"  ✓ Using SMILES-based drug embedding (768 dim) instead of drug name vocabulary")
    
    # Filter predictor_df to only include metrics the model was trained on
    # The predict() method will handle missing metrics gracefully, but we should filter
    model_metrics = set(time_series_predictor.metric_names)
    # Include dosage if model supports it, otherwise just use standard features
    required_cols = ['patient_id', 'drug_name', 'age', 'sex', 'bmi', 'adherence', 'days_on_drug']
    if 'dosage' in predictor_df.columns:
        required_cols.append('dosage')
    filtered_predictor_df = predictor_df[required_cols].copy()
    
    # Add baseline and delta columns only for metrics the model knows
    for metric in model_metrics:
        base_name = metric.replace('_delta', '')
        baseline_col = f'{base_name}_baseline'
        delta_col = f'{base_name}_delta'
        
        if baseline_col in predictor_df.columns:
            filtered_predictor_df[baseline_col] = predictor_df[baseline_col]
        else:
            filtered_predictor_df[baseline_col] = 0.0
        
        if delta_col in predictor_df.columns:
            filtered_predictor_df[delta_col] = predictor_df[delta_col]
        else:
            filtered_predictor_df[delta_col] = 0.0
    
    predictor_df = filtered_predictor_df
    
    # Convert static predictions (from PharmacodynamicPredictor) to time series
    # This model ONLY converts static -> temporal, it does NOT predict drug effects
    # Model uses day-based timepoints, we'll convert to seconds
    # Pass SMILES-based drug embedding directly (not drug name vocabulary)
    time_series_df = time_series_predictor.predict(
        predictor_df,
        return_uncertainty=False,
        drug_embedding=emb  # Pass 768-dim SMILES-based embedding
    )
    
    # Get model's trained timepoints (in days)
    model_timepoints_days = time_series_predictor.config.timepoints
    
    # Convert model's day-based predictions to second-based predictions
    # Model outputs deltas at day timepoints, we need to interpolate to second timepoints
    print(f"  ⚠ Model was trained on day-based timepoints: {model_timepoints_days}")
    print(f"    Converting to second-based timepoints: {len(timepoints_seconds)} timepoints")
    print(f"    Interval: every {interval_seconds} seconds up to {total_seconds} seconds ({total_hours} hours)")
    
    # Build all data at once to avoid DataFrame fragmentation
    # Collect all interpolated values first, then create DataFrame columns all at once
    new_columns_data = {}  # {col_name: [values for each row]}
    
    # Initialize all new columns with NaN
    for seconds in timepoints_seconds:
        if seconds == 0:
            continue  # Skip baseline
        col_name = f'{seconds}sec'
        new_columns_data[col_name] = [np.nan] * len(time_series_df)
    
    # Interpolate from days to seconds for each metric
    for metric_idx, metric_name in enumerate(time_series_df['metric_name'].unique()):
        metric_mask = time_series_df['metric_name'] == metric_name
        metric_rows = time_series_df[metric_mask].copy()
        
        if len(metric_rows) == 0:
            continue
        
        # Get baseline and final_delta
        baseline = metric_rows['baseline'].iloc[0]
        final_delta = metric_rows['final_delta'].iloc[0]
        
        # Collect model predictions at trained day timepoints (convert to seconds for interpolation)
        model_predictions_seconds = {}
        for day in model_timepoints_days:
            col_name = f'day_{day}'
            if col_name in metric_rows.columns:
                # Model outputs delta at this day
                delta_at_day = metric_rows[col_name].iloc[0]
                # Convert day to seconds (for model interpolation)
                # Model uses days, but we're simulating 3 hours, so we scale proportionally
                # For single injection, we map model's day predictions to 3-hour window
                seconds_at_day = int((day / 180.0) * total_seconds)  # Scale 180 days to 3 hours
                model_predictions_seconds[seconds_at_day] = delta_at_day
        
        # Add baseline (0 seconds, delta = 0) and final (total_seconds, final_delta)
        model_predictions_seconds[0] = 0.0  # Baseline delta is 0
        model_predictions_seconds[total_seconds] = final_delta  # Final delta at total seconds
        
        # Sort timepoints for interpolation
        sorted_model_seconds = sorted(model_predictions_seconds.keys())
        sorted_model_values = [model_predictions_seconds[s] for s in sorted_model_seconds]
        
        # Interpolate to requested second timepoints
        # Create interpolation function (linear interpolation)
        interp_func = interp1d(
            sorted_model_seconds, 
            sorted_model_values, 
            kind='linear',
            fill_value='extrapolate',  # Extrapolate beyond trained range
            bounds_error=False
        )
        
        # Predict at requested second timepoints and store in dictionary
        for seconds in timepoints_seconds:
            if seconds == 0:
                continue  # Skip baseline (already in baseline column)
            
            col_name = f'{seconds}sec'
            # Interpolate delta at this second
            interpolated_delta = float(interp_func(seconds))
            # Convert to absolute value (baseline + delta)
            absolute_value = baseline + interpolated_delta
            
            # Store value for all rows with this metric
            for row_idx in time_series_df[metric_mask].index:
                new_columns_data[col_name][row_idx] = absolute_value
    
    # Add all new columns at once using pd.concat (much faster!)
    if new_columns_data:
        new_df = pd.DataFrame(new_columns_data, index=time_series_df.index)
        time_series_df = pd.concat([time_series_df, new_df], axis=1)
    
    # Remove old day-based columns
    for day in model_timepoints_days:
        col_name = f'day_{day}'
        if col_name in time_series_df.columns:
            time_series_df.drop(columns=[col_name], inplace=True, errors='ignore')
    
    # Recalculate final_delta as (final_second_value - baseline) for each metric
    final_second_col = f'{total_seconds}sec'
    if final_second_col in time_series_df.columns:
        # final_delta = final_second_value - baseline
        time_series_df['final_delta'] = time_series_df[final_second_col] - time_series_df['baseline']
    else:
        # If final second column doesn't exist, use the last available timepoint
        last_second = max([s for s in timepoints_seconds if s <= total_seconds], default=None)
        if last_second is not None:
            last_second_col = f'{last_second}sec'
            if last_second_col in time_series_df.columns:
                time_series_df['final_delta'] = time_series_df[last_second_col] - time_series_df['baseline']
            else:
                print(f"  ⚠ Warning: Could not find final second column, keeping original final_delta")
        else:
            print(f"  ⚠ Warning: No valid timepoints found, keeping original final_delta")
    
    print(f"  ✓ Generated trajectories for {len(time_series_df)} metric-patient combinations")
    print(f"    Time points: {len(timepoints_seconds)} timepoints in seconds (every {interval_seconds} sec up to {total_seconds} sec = {total_hours} hours)")
    print(f"    Note: Time Series Predictor converts static predictions to temporal trajectories")
    print(f"    Note: final_delta = {total_seconds}sec - baseline")
    
    # Step 5: Save results
    print("\n[5/5] Saving results...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped folder with presentable format: "2025-12-14_01h13m"
    timestamp = time.strftime("%Y-%m-%d_%Hh%Mm")
    run_folder = OUTPUT_DIR / timestamp
    run_folder.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Created run folder: {run_folder.name}")
    
    # Save time series data
    out_file = run_folder / 'time_series_simulation.csv'
    time_series_df.to_csv(out_file, index=False, encoding='utf-8')
    print(f"  ✓ Time series saved to: {out_file}")
    
    # Also save a summary with key metrics
    summary_file = run_folder / 'summary_key_metrics.csv'
    key_metrics_summary = time_series_df[time_series_df['metric_name'].isin(key_metrics)]
    if len(key_metrics_summary) > 0:
        key_metrics_summary.to_csv(summary_file, index=False, encoding='utf-8')
        print(f"  ✓ Summary (key metrics) saved to: {summary_file}")
    
    # Generate feature/time graphs
    print("\n[6/6] Generating feature/time graphs...")
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        # Get all timepoint columns (those ending with 'sec')
        timepoint_cols = [col for col in time_series_df.columns if col.endswith('sec')]
        timepoint_cols.sort(key=lambda x: int(x.replace('sec', '')))
        
        # Extract time values in seconds and convert to hours for x-axis
        time_values = [int(col.replace('sec', '')) for col in timepoint_cols]
        time_hours = [t / SECONDS_PER_HOUR for t in time_values]  # Convert to hours
        
        # Get all unique metrics
        all_metrics = time_series_df['metric_name'].unique()
        
        # Create individual graphs for each feature
        graphs_dir = run_folder / 'graphs'
        graphs_dir.mkdir(exist_ok=True)
        
        graphs_created = 0
        for metric in all_metrics:
            metric_data = time_series_df[time_series_df['metric_name'] == metric].iloc[0]
            baseline = metric_data['baseline']
            
            # Extract values over time
            values = []
            for col in timepoint_cols:
                if col in metric_data:
                    values.append(metric_data[col])
                else:
                    values.append(np.nan)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot trajectory
            ax.plot(time_hours, values, 'b-', linewidth=2, label=f'{metric}')
            ax.axhline(y=baseline, color='r', linestyle='--', linewidth=1.5, label='Baseline')
            
            # Formatting
            ax.set_xlabel('Time (hours)', fontsize=12)
            ax.set_ylabel(f'{metric} Value', fontsize=12)
            ax.set_title(f'{metric} Over Time (Single Injection: {dosage} mg)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Save individual graph
            graph_file = graphs_dir / f'{metric}_over_time.png'
            plt.savefig(graph_file, dpi=150, bbox_inches='tight')
            plt.close()
            graphs_created += 1
        
        # Create summary multi-panel figure (all features in one)
        n_metrics = len(all_metrics)
        n_cols = 4
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for idx, metric in enumerate(all_metrics):
            ax = axes[idx]
            metric_data = time_series_df[time_series_df['metric_name'] == metric].iloc[0]
            baseline = metric_data['baseline']
            
            # Extract values over time
            values = []
            for col in timepoint_cols:
                if col in metric_data:
                    values.append(metric_data[col])
                else:
                    values.append(np.nan)
            
            # Plot
            ax.plot(time_hours, values, 'b-', linewidth=1.5)
            ax.axhline(y=baseline, color='r', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_title(metric, fontsize=10)
            ax.set_xlabel('Time (hours)', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'All Features Over Time (Single Injection: {dosage} mg)', 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save summary graph in graphs folder
        summary_graph_file = graphs_dir / 'all_features_over_time.png'
        plt.savefig(summary_graph_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Generated {graphs_created} individual feature graphs")
        print(f"  ✓ Generated summary multi-panel graph")
        print(f"    All graphs saved in: {graphs_dir}")
        print(f"    Individual graphs: {graphs_created} files")
        print(f"    Summary graph: all_features_over_time.png")
        
    except ImportError:
        print(f"  ⚠ Warning: matplotlib not available, skipping graph generation")
    except Exception as e:
        print(f"  ⚠ Warning: Failed to generate graphs: {e}")
    
    # Save metadata
    import json
    metrics = {
        'patient': {
            'age': int(age),
            'sex': str(sex),
            'height_cm': float(height),
            'weight_kg': float(weight),
            'bmi': float(S0['BMXBMI'].values[0]),
            'adherence': float(adherence),
            'total_hours': float(total_hours),
        },
        'drug': {
            'smiles': str(smiles),
            'drug_name': str(drug_name),
            'embedding_dim': int(emb.shape[0]),
            'embedding_norm': float(np.linalg.norm(emb)),
        },
        'time_series': {
            'timepoints_seconds': timepoints_seconds,
            'interval_seconds': interval_seconds,
            'total_seconds': total_seconds,
            'total_hours': float(total_hours),
            'num_metrics': len(time_series_df),
            'metrics': list(time_series_df['metric_name'].unique()),
        },
        'dosage_mg': float(dosage),
        'run_info': {
            'timestamp': timestamp,
            'run_folder': str(run_folder),
        }
    }
    
    metrics_file = run_folder / 'metrics.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  ✓ Metrics saved to: {metrics_file}")
    
    print(f"\n  📁 All results saved in: {run_folder}")
    
    # Print summary
    print("\n" + "="*70)
    print("TIME SERIES PREDICTION COMPLETE")
    print("="*70)
    print(f"Patient: {age}yo {sex}, BMI={S0['BMXBMI'].values[0]:.1f}")
    print(f"Drug: {drug_name} ({smiles[:50]}...), Dosage: {dosage} mg")
    print(f"Duration: {total_hours} hours ({total_seconds} seconds)")
    print(f"\nTime Series Shape: {time_series_df.shape}")
    print(f"Metrics predicted: {len(time_series_df['metric_name'].unique())}")
    print(f"Time points: {len(timepoints_seconds)} timepoints (every {interval_seconds} sec up to {total_seconds} sec)")
    print(f"\n📁 Results folder: {run_folder}")
    print("\nSample trajectories (first 5 metrics):")
    # Show sample columns (baseline, final_delta, and a few timepoints)
    sample_cols = ['metric_name', 'baseline', 'final_delta']
    # Add a few timepoint columns if they exist
    for seconds in timepoints_seconds[:5]:  # Show first 5 timepoints
        col = f'{seconds}sec'
        if col in time_series_df.columns:
            sample_cols.append(col)
    print(time_series_df[sample_cols].head())
    print("="*70 + "\n")
    
    return time_series_df


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments()
    
    # Get user inputs (unless --no-input flag is set or all args provided)
    if args.no_input or (args.smiles and args.age and args.sex and args.height and args.weight):
        # Use command-line args or defaults
        smiles = args.smiles if args.smiles else 'CC(=O)OC1=CC=CC=C1C(=O)O'
        drug_name = args.drug_name if args.drug_name else 'Unknown'
        age = args.age if args.age else 55
        sex = args.sex if args.sex else 'M'
        height = args.height if args.height else 175.0
        weight = args.weight if args.weight else 80.0
        adherence = args.adherence if args.adherence else 1.0
        dosage = args.dosage if args.dosage else 10.0
        try:
            total_hours = parse_time_to_seconds(args.total_time) / 3600.0
            interval_seconds = int(round(parse_time_to_seconds(args.interval)))
        except ValueError as e:
            print(f"ERROR: Invalid time format. {e}")
            print(TIME_FORMAT_HELP)
            sys.exit(1)
        if interval_seconds < 1:
            print("ERROR: --interval must be at least 1 second")
            sys.exit(1)
    else:
        # Get interactive input
        user_inputs = get_user_inputs()
        age = user_inputs['patient']['age']
        sex = user_inputs['patient']['sex']
        height = user_inputs['patient']['height']
        weight = user_inputs['patient']['weight']
        adherence = user_inputs['patient']['adherence']
        dosage = user_inputs['patient']['dosage']
        smiles = user_inputs['drug']['smiles']
        drug_name = user_inputs['drug']['drug_name']
        total_hours = user_inputs.get('total_hours', 3.0)
        interval_seconds = user_inputs.get('interval_seconds', 10)
    
    # Validate inputs
    if not smiles:
        print("ERROR: Drug SMILES required")
        sys.exit(1)
    
    if sex not in ['M', 'F']:
        print("ERROR: Sex must be 'M' or 'F'")
        sys.exit(1)
    
    # Run simulation (single injection; final time and checkpoint from inputs)
    time_series = run(
        smiles=smiles,
        drug_name=drug_name,
        age=age,
        sex=sex,
        height=height,
        weight=weight,
        adherence=adherence,
        dosage=dosage,
        timepoints_seconds=None,
        interval_seconds=interval_seconds,
        total_hours=total_hours,
    )
