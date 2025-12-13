"""
Generate realistic demo drug-response dataset based on known pharmacology.

This creates a COMPREHENSIVE synthetic dataset with:
- Realistic baseline labs with demographic dependencies
- Drug effects with temporal dynamics and adherence
- Adverse events
- Multiple dataset variants
- Ground truth metadata for benchmarking
- Comprehensive validation

Based on published clinical trial data and drug mechanisms.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'n_patients': 2000,  # Increased for more diversity
    'n_samples': 10000,  # Increased to 10,000 samples
    'individual_cv': 0.20,  # Coefficient of variation for individual response
    'min_adherence': 0.70,
    'max_adherence': 1.00,
    'adverse_event_rate': 0.10,
    'missing_rate': 0.05,
    'measurement_error_cv': 0.03,
    'outlier_rate': 0.005,
}

# Physiological bounds for lab values (min, max)
LAB_BOUNDS = {
    'LBXSGL': (70, 300),      # Glucose (mg/dL)
    'LBXGH': (4.0, 12.0),     # HbA1c (%)
    'LBXTC': (100, 400),      # Total cholesterol (mg/dL)
    'LBDLDL': (40, 300),      # LDL (mg/dL)
    'LBXSCH': (20, 100),      # HDL (mg/dL)
    'LBDSCR': (40, 200),      # Creatinine (μmol/L or mg/dL equivalent)
    'LBXSBU': (5, 50),        # BUN (mg/dL)
    'LBXSK': (3.0, 6.0),      # Potassium (mmol/L)
    'LBXSNA': (130, 150),     # Sodium (mmol/L)
    'LBXSCA': (8.0, 11.5),    # Calcium (mg/dL)
    'LBXSAS': (10, 100),      # AST (U/L)
    'LBXSAL': (10, 100),      # ALT (U/L)
    'LBXSGB': (0.2, 3.0),     # Bilirubin (mg/dL)
    'LBXSUA': (2.0, 10.0),    # Uric acid (mg/dL)
    'BPXSY1': (90, 200),      # Systolic BP (mmHg)
    'BPXDI1': (50, 120),      # Diastolic BP (mmHg)
}

# Labs with percentage-based effects (scale with baseline)
PERCENTAGE_BASED_LABS = {'LBXSGL', 'LBXGH', 'LBXTC', 'LBDLDL'}

# Known drug effects from literature (FDA labels, meta-analyses)
DRUG_EFFECTS = {
    'Metformin': {
        'LBXSGL': -25.0,      # Glucose: -20 to -30 mg/dL
        'LBXGH': -1.0,        # HbA1c: -0.8 to -1.2%
        'LBXTC': -5.0,        # Cholesterol: slight decrease
        'LBDSCR': 2.0,        # Creatinine: slight increase
        'tau_days': 60,       # Time to steady state
    },
    'Atorvastatin': {
        'LBXTC': -50.0,       # Total cholesterol: -40 to -60 mg/dL
        'LBDLDL': -60.0,      # LDL: -50 to -70 mg/dL  
        'LBXSCH': 5.0,        # HDL: +5 to +10 mg/dL
        'LBXSAS': 10.0,       # AST: slight increase
        'LBXSGL': 3.0,        # Glucose: slight increase
        'tau_days': 30,       # Time to steady state
    },
    'Lisinopril': {
        'BPXSY1': -12.0,      # Systolic BP: -10 to -15 mmHg
        'BPXDI1': -8.0,       # Diastolic BP: -5 to -10 mmHg
        'LBDSCR': 5.0,        # Creatinine: increase (reduced GFR)
        'LBXSK': 0.3,         # Potassium: increase (retention)
        'LBXSBU': 3.0,        # BUN: increase
        'tau_days': 14,       # Time to steady state
    },
    'Amlodipine': {
        'BPXSY1': -10.0,      # Systolic BP
        'BPXDI1': -7.0,       # Diastolic BP
        'LBDSCR': 0.5,        # Minimal kidney effect
        'tau_days': 7,        # Time to steady state
    },
    'Furosemide': {
        'BPXSY1': -8.0,       # BP decrease
        'LBXSK': -0.5,        # Potassium: decrease (wasting)
        'LBXSNA': -2.0,       # Sodium: decrease
        'LBDSCR': 3.0,        # Creatinine: increase
        'LBXSBU': 5.0,        # BUN: increase
        'tau_days': 1,        # Time to steady state (fast acting)
    },
    'Simvastatin': {
        'LBXTC': -45.0,       # Similar to atorvastatin
        'LBDLDL': -55.0,
        'LBXSCH': 8.0,
        'LBXSAS': 12.0,
        'tau_days': 30,
    },
    'Prednisone': {
        'LBXSGL': 30.0,       # Glucose: increase (insulin resistance)
        'LBXSK': -0.4,        # Potassium: decrease
        'LBXSNA': 3.0,        # Sodium: increase (retention)
        'LBXTC': 15.0,        # Cholesterol: increase
        'tau_days': 3,        # Time to steady state
    },
    'Ibuprofen': {
        'LBDSCR': 8.0,        # Creatinine: increase (nephrotoxic)
        'LBXSBU': 6.0,        # BUN: increase
        'LBXSK': 0.2,         # Potassium: slight increase
        'tau_days': 1,        # Time to steady state
    },
    'Aspirin': {
        'LBDSCR': 1.0,        # Minimal kidney effect at low dose
        'LBXSUA': -0.5,       # Uric acid: slight decrease
        'tau_days': 1,
    },
    'Allopurinol': {
        'LBXSUA': -3.0,       # Uric acid: decrease (xanthine oxidase inhibitor)
        'LBDSCR': 1.5,        # Creatinine: slight increase
        'tau_days': 30,       # Time to steady state
    },
    # Additional drugs to reach 50+
    'Rosuvastatin': {
        'LBXTC': -55.0,       'LBDLDL': -65.0,      'LBXSCH': 10.0,       'LBXSAS': 8.0,        'tau_days': 28,
    },
    'Pravastatin': {
        'LBXTC': -40.0,       'LBDLDL': -50.0,      'LBXSCH': 6.0,        'tau_days': 30,
    },
    'Enalapril': {
        'BPXSY1': -11.0,      'BPXDI1': -7.0,       'LBDSCR': 4.0,        'LBXSK': 0.25,        'tau_days': 14,
    },
    'Ramipril': {
        'BPXSY1': -10.0,      'BPXDI1': -6.0,       'LBDSCR': 3.5,        'LBXSK': 0.2,         'tau_days': 14,
    },
    'Losartan': {
        'BPXSY1': -9.0,       'BPXDI1': -6.0,       'LBDSCR': 2.0,        'LBXSK': 0.1,         'tau_days': 14,
    },
    'Valsartan': {
        'BPXSY1': -9.5,       'BPXDI1': -6.5,       'LBDSCR': 2.5,        'tau_days': 14,
    },
    'Hydrochlorothiazide': {
        'BPXSY1': -8.0,       'BPXDI1': -5.0,       'LBXSK': -0.4,        'LBXSNA': -1.5,       'LBXSUA': 1.0,        'tau_days': 7,
    },
    'Chlorthalidone': {
        'BPXSY1': -9.0,       'BPXDI1': -6.0,       'LBXSK': -0.5,        'LBXSUA': 1.2,        'tau_days': 7,
    },
    'Atenolol': {
        'BPXSY1': -7.0,       'BPXDI1': -5.0,       'LBXSGL': 5.0,        'LBXTC': 8.0,         'tau_days': 7,
    },
    'Metoprolol': {
        'BPXSY1': -7.5,       'BPXDI1': -5.5,       'LBXSGL': 4.0,        'tau_days': 7,
    },
    'Carvedilol': {
        'BPXSY1': -8.0,       'BPXDI1': -6.0,       'tau_days': 7,
    },
    'Propranolol': {
        'BPXSY1': -6.0,       'BPXDI1': -4.0,       'LBXTC': 10.0,        'tau_days': 7,
    },
    'Diltiazem': {
        'BPXSY1': -9.0,       'BPXDI1': -6.0,       'tau_days': 7,
    },
    'Verapamil': {
        'BPXSY1': -8.5,       'BPXDI1': -6.5,       'LBDSCR': 1.0,        'tau_days': 7,
    },
    'Nifedipine': {
        'BPXSY1': -10.0,       'BPXDI1': -7.0,       'LBDSCR': 0.5,        'tau_days': 7,
    },
    'Glipizide': {
        'LBXSGL': -30.0,      'LBXGH': -1.2,        'LBXTC': 3.0,         'tau_days': 7,
    },
    'Glyburide': {
        'LBXSGL': -32.0,      'LBXGH': -1.3,        'LBXTC': 4.0,         'tau_days': 7,
    },
    'Pioglitazone': {
        'LBXSGL': -20.0,      'LBXGH': -0.8,        'LBXTC': 8.0,         'LBXSCH': 5.0,        'tau_days': 60,
    },
    'Sitagliptin': {
        'LBXSGL': -18.0,      'LBXGH': -0.7,        'tau_days': 14,
    },
    'Empagliflozin': {
        'LBXSGL': -22.0,      'LBXGH': -0.9,        'BPXSY1': -3.0,       'LBDSCR': 2.0,        'tau_days': 14,
    },
    'Canagliflozin': {
        'LBXSGL': -24.0,      'LBXGH': -1.0,        'BPXSY1': -3.5,       'LBDSCR': 2.5,        'tau_days': 14,
    },
    'Warfarin': {
        'LBXSGB': 0.3,        'LBXSAS': 5.0,        'tau_days': 7,
    },
    'Digoxin': {
        'LBXSK': 0.1,         'LBDSCR': 1.0,        'tau_days': 7,
    },
    'Spironolactone': {
        'BPXSY1': -4.0,       'LBXSK': 0.5,         'LBXSNA': -1.0,       'tau_days': 14,
    },
    'Eplerenone': {
        'BPXSY1': -3.5,       'LBXSK': 0.4,         'tau_days': 14,
    },
    'Amiodarone': {
        'LBXSAS': 15.0,       'LBXSAL': 12.0,       'LBXTC': 10.0,        'tau_days': 30,
    },
    'Flecainide': {
        'LBXSK': 0.1,         'tau_days': 7,
    },
    'Sotalol': {
        'BPXSY1': -5.0,       'LBXSK': 0.2,         'tau_days': 7,
    },
    'Dofetilide': {
        'LBXSK': 0.15,        'tau_days': 7,
    },
    'Amoxicillin': {
        'LBXSAS': 3.0,        'LBXSAL': 2.0,        'tau_days': 1,
    },
    'Ciprofloxacin': {
        'LBXSK': 0.1,         'tau_days': 1,
    },
    'Levofloxacin': {
        'LBXSK': 0.1,         'LBXSGL': 2.0,        'tau_days': 1,
    },
    'Azithromycin': {
        'LBXSAS': 2.0,        'tau_days': 1,
    },
    'Ceftriaxone': {
        'LBXSGB': 0.2,        'tau_days': 1,
    },
    'Vancomycin': {
        'LBDSCR': 5.0,        'LBXSBU': 4.0,        'tau_days': 3,
    },
    'Gentamicin': {
        'LBDSCR': 8.0,        'LBXSBU': 6.0,        'tau_days': 1,
    },
    'Cyclosporine': {
        'LBDSCR': 12.0,       'LBXSBU': 8.0,        'LBXTC': 20.0,        'LBXSGL': 8.0,        'tau_days': 14,
    },
    'Tacrolimus': {
        'LBDSCR': 10.0,       'LBXSBU': 7.0,        'LBXTC': 15.0,        'LBXSGL': 6.0,        'tau_days': 14,
    },
    'Methotrexate': {
        'LBXSAS': 8.0,        'LBXSAL': 6.0,        'LBDSCR': 2.0,        'tau_days': 7,
    },
    'Sulfasalazine': {
        'LBXSAS': 5.0,        'LBXSAL': 4.0,        'tau_days': 7,
    },
    'Hydroxychloroquine': {
        'LBXSAS': 3.0,        'tau_days': 14,
    },
    'Colchicine': {
        'LBDSCR': 1.5,        'tau_days': 1,
    },
    'Probenecid': {
        'LBXSUA': -2.0,       'tau_days': 7,
    },
    'Fenofibrate': {
        'LBXTC': -25.0,       'LBDLDL': -15.0,      'LBXSCH': 12.0,       'LBXSUA': -1.5,       'tau_days': 30,
    },
    'Gemfibrozil': {
        'LBXTC': -20.0,       'LBDLDL': -12.0,      'LBXSCH': 10.0,       'tau_days': 30,
    },
    'Ezetimibe': {
        'LBXTC': -15.0,       'LBDLDL': -18.0,      'tau_days': 14,
    },
    'Niacin': {
        'LBXTC': -10.0,       'LBDLDL': -8.0,       'LBXSCH': 15.0,       'LBXSGL': 5.0,        'tau_days': 30,
    },
}

# Adverse event definitions
ADVERSE_EVENTS = {
    'Atorvastatin': {'statin_myalgia': 0.10},
    'Simvastatin': {'statin_myalgia': 0.12},
    'Lisinopril': {'ace_cough': 0.15},
    'Metformin': {'metformin_gi_upset': 0.15},
}


# ============================================================================
# PATIENT GENERATION
# ============================================================================

def generate_realistic_patient_demographics(n_patients: int = None) -> pd.DataFrame:
    """
    Generate realistic patient demographics.
    
    Args:
        n_patients: Number of patients to generate (uses CONFIG if None)
        
    Returns:
        DataFrame with patient demographics
    """
    if n_patients is None:
        n_patients = CONFIG['n_patients']
    
    patients = {
        'patient_id': range(n_patients),
        'AGE': np.random.normal(55, 15, n_patients).clip(18, 90).astype(int),
        'SEX': np.random.choice(['M', 'F'], n_patients),
        'BMXHT': np.random.normal(170, 10, n_patients).clip(150, 200),
        'BMXWT': np.random.normal(80, 18, n_patients).clip(45, 150),
    }
    
    # Calculate BMI
    patients['BMXBMI'] = patients['BMXWT'] / (patients['BMXHT'] / 100) ** 2
    
    return pd.DataFrame(patients)


def generate_baseline_labs(patients_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate realistic baseline lab values with age/sex/BMI dependencies.
    
    Based on NHANES population data. Slightly elevated values to represent
    patients who need medication.
    
    References:
    - NHANES 2017-2018 Lab Data
    - Considers age-related increase in glucose (~0.5 mg/dL per year)
    - Sex differences in creatinine (men 20% higher muscle mass)
    
    Args:
        patients_df: DataFrame with patient demographics
        
    Returns:
        DataFrame with baseline lab values
    """
    n = len(patients_df)
    
    # Base baseline labs (slightly unhealthy population - why they need drugs)
    baseline_labs = {}
    
    # Generate base values
    baseline_labs['LBXSGL'] = np.random.normal(110, 25, n)  # Glucose
    baseline_labs['LBXGH'] = np.random.normal(6.0, 1.0, n)  # HbA1c
    baseline_labs['LBXTC'] = np.random.normal(210, 35, n)  # Cholesterol
    baseline_labs['LBDLDL'] = np.random.normal(130, 30, n)  # LDL
    baseline_labs['LBXSCH'] = np.random.normal(50, 12, n)  # HDL
    baseline_labs['LBDSCR'] = np.random.normal(90, 20, n)  # Creatinine
    baseline_labs['LBXSBU'] = np.random.normal(15, 5, n)  # BUN
    baseline_labs['LBXSK'] = np.random.normal(4.0, 0.4, n)  # Potassium
    baseline_labs['LBXSNA'] = np.random.normal(140, 3, n)  # Sodium
    baseline_labs['LBXSCA'] = np.random.normal(9.5, 0.5, n)  # Calcium
    baseline_labs['LBXSAS'] = np.random.normal(25, 10, n)  # AST
    baseline_labs['LBXSAL'] = np.random.normal(28, 12, n)  # ALT
    baseline_labs['LBXSGB'] = np.random.normal(0.8, 0.3, n)  # Bilirubin
    baseline_labs['LBXSUA'] = np.random.normal(5.5, 1.2, n)  # Uric acid
    baseline_labs['BPXSY1'] = np.random.normal(135, 18, n)  # Systolic BP
    baseline_labs['BPXDI1'] = np.random.normal(85, 12, n)  # Diastolic BP
    
    # Convert to DataFrame for easier manipulation
    baseline_df = pd.DataFrame(baseline_labs)
    
    # Apply demographic dependencies
    # Age effects
    age_centered = patients_df['AGE'] - 55
    baseline_df['LBXSGL'] += 0.5 * age_centered  # Glucose increases with age
    baseline_df['LBXTC'] += 0.8 * age_centered  # Cholesterol increases with age
    baseline_df['LBDLDL'] += 0.6 * age_centered  # LDL increases with age
    baseline_df['BPXSY1'] += 0.3 * age_centered  # BP increases with age
    
    # Sex effects
    is_male = (patients_df['SEX'] == 'M').astype(float)
    baseline_df['LBDSCR'] *= (1.0 + 0.2 * is_male)  # Men have higher creatinine
    baseline_df['LBXSCH'] *= (1.0 - 0.1 * is_male)  # Men have slightly lower HDL
    
    # BMI effects
    bmi_centered = patients_df['BMXBMI'] - 25
    baseline_df['LBXSGL'] += np.where(bmi_centered > 0, 2.0 * bmi_centered, 0)  # Obesity increases glucose
    baseline_df['LBXTC'] += np.where(bmi_centered > 0, 1.5 * bmi_centered, 0)  # Obesity increases cholesterol
    baseline_df['BPXSY1'] += np.where(bmi_centered > 0, 0.5 * bmi_centered, 0)  # Obesity increases BP
    
    # Clip to physiological bounds
    for lab, bounds in LAB_BOUNDS.items():
        if lab in baseline_df.columns:
            baseline_df[lab] = baseline_df[lab].clip(bounds[0], bounds[1])
    
    # Ensure all values are positive
    baseline_df = baseline_df.clip(lower=0.1)
    
    return baseline_df


# ============================================================================
# DRUG RESPONSE GENERATION (SIMPLIFIED)
# ============================================================================

def generate_drug_responses(
    patients_df: pd.DataFrame,
    baseline_labs_df: pd.DataFrame,
    n_samples: int = None
) -> pd.DataFrame:
    """
    Generate drug-response pairs with realistic effects.
    
    Simplified 3-step approach:
    1. Base effect from literature
    2. Individual variability (CV = 20%)
    3. Clip to physiological bounds
    
    Args:
        patients_df: Patient demographics
        baseline_labs_df: Baseline lab values
        n_samples: Number of samples to generate (uses CONFIG if None)
        
    Returns:
        DataFrame with drug-response pairs
    """
    if n_samples is None:
        n_samples = CONFIG['n_samples']
    
    drug_names = [d for d in DRUG_EFFECTS.keys() if d != 'tau_days']
    
    samples = []
    
    for _ in range(n_samples):
        # Random patient
        patient_idx = np.random.randint(0, len(patients_df))
        patient = patients_df.iloc[patient_idx]
        baseline_labs = baseline_labs_df.iloc[patient_idx]
        
        # Random drug
        drug_name = np.random.choice(drug_names)
        drug_effects = {k: v for k, v in DRUG_EFFECTS[drug_name].items() if k != 'tau_days'}
        tau_days = DRUG_EFFECTS[drug_name].get('tau_days', 30)
        
        # Random days on drug (1-180)
        days_on_drug = np.random.randint(1, 181)
        
        # Adherence (70-100%)
        adherence = np.random.uniform(CONFIG['min_adherence'], CONFIG['max_adherence'])
        
        # Compute response
        response = {}
        
        for lab, mean_effect in drug_effects.items():
            if lab not in baseline_labs.index:
                continue
            
            baseline_val = baseline_labs[lab]
            
            # Step 1: Base effect from literature
            base_effect = mean_effect
            
            # Step 2: Individual variability (CV = 20%)
            individual_effect = base_effect * np.random.normal(1.0, CONFIG['individual_cv'])
            
            # Apply temporal dynamics (drug effect builds up over time)
            time_factor = 1.0 - np.exp(-days_on_drug / tau_days)
            individual_effect *= time_factor
            
            # Apply adherence
            individual_effect *= adherence
            
            # Step 3: Clip to physiological bounds
            final_val = baseline_val + individual_effect
            bounds = LAB_BOUNDS.get(lab, (0, np.inf))
            final_val = np.clip(final_val, bounds[0], bounds[1])
            effect = final_val - baseline_val
            
            response[f'{lab}_delta'] = effect
            response[f'{lab}_baseline'] = baseline_val
        
        # Add metadata
        sample = {
            'patient_id': patient['patient_id'],
            'drug_name': drug_name,
            'age': patient['AGE'],
            'sex': patient['SEX'],
            'bmi': patient['BMXBMI'],
            'days_on_drug': days_on_drug,
            'adherence': adherence,
            **response
        }
        
        # Add adverse events
        if drug_name in ADVERSE_EVENTS:
            for event_name, event_prob in ADVERSE_EVENTS[drug_name].items():
                sample[event_name] = 1 if np.random.random() < event_prob else 0
        else:
            # Initialize to 0 for drugs without known adverse events
            for event_name in ['statin_myalgia', 'ace_cough', 'metformin_gi_upset']:
                if event_name not in sample:
                    sample[event_name] = 0
        
        samples.append(sample)
    
    return pd.DataFrame(samples)


# ============================================================================
# DATA QUALITY ISSUES
# ============================================================================

def add_realistic_data_quality_issues(
    df: pd.DataFrame,
    missing_rate: float = None,
    measurement_error_cv: float = None
) -> pd.DataFrame:
    """
    Add realistic data quality issues that occur in real clinical data.
    
    - Missing values with MAR (Missing At Random) mechanism
    - Measurement error/lab variability
    - Occasional outliers from lab errors
    
    Args:
        df: Input DataFrame
        missing_rate: Rate of missing values (uses CONFIG if None)
        measurement_error_cv: Coefficient of variation for measurement error
        
    Returns:
        DataFrame with data quality issues added
    """
    if missing_rate is None:
        missing_rate = CONFIG['missing_rate']
    if measurement_error_cv is None:
        measurement_error_cv = CONFIG['measurement_error_cv']
    
    df_noisy = df.copy()
    
    # 1. Missing values - more likely for certain labs and older patients
    for col in [c for c in df.columns if c.endswith('_baseline') or c.endswith('_delta')]:
        # Base missing rate
        missing_prob = missing_rate
        
        # Increase missingness for elderly (harder to get blood samples)
        if 'age' in df.columns:
            age_factor = np.clip((df['age'] - 50) / 100, 0, 0.05)
            missing_prob = missing_rate + age_factor
        
        # Apply missingness
        mask = np.random.random(len(df)) < missing_prob
        df_noisy.loc[mask, col] = np.nan
    
    # 2. Measurement error (lab-to-lab variability)
    for col in [c for c in df.columns if c.endswith('_baseline')]:
        non_missing = df_noisy[col].notna()
        if non_missing.any():
            measurement_noise = np.random.normal(1.0, measurement_error_cv, non_missing.sum())
            df_noisy.loc[non_missing, col] *= measurement_noise
    
    # 3. Rare outliers (0.5% - data entry errors, lab errors)
    for col in [c for c in df.columns if c.endswith('_baseline') or c.endswith('_delta')]:
        outlier_mask = np.random.random(len(df)) < CONFIG['outlier_rate']
        if outlier_mask.any():
            # Replace with extreme values
            df_noisy.loc[outlier_mask, col] *= np.random.choice([0.3, 3.0], outlier_mask.sum())
    
    # 4. Add data quality score
    df_noisy['data_quality_score'] = np.random.choice(
        ['high', 'medium', 'low'],
        size=len(df),
        p=[0.85, 0.12, 0.03]
    )
    
    return df_noisy


# ============================================================================
# VALIDATION
# ============================================================================

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate dataset quality.
    
    IMPORTANT: Deltas are CHANGES, not absolute values, so they can be negative
    or outside bounds. We only validate:
    1. Baseline values are within bounds
    2. Final values (baseline + delta) are within bounds
    3. No negative baseline values
    
    Args:
        df: DataFrame to validate
        
    Returns:
        (is_valid, error_list) tuple
    """
    errors = []
    
    # Check baseline values are within bounds (deltas are changes, not absolute values!)
    for lab, bounds in LAB_BOUNDS.items():
        baseline_col = f'{lab}_baseline'
        if baseline_col in df.columns:
            violations = df[baseline_col].notna() & ((df[baseline_col] < bounds[0]) | (df[baseline_col] > bounds[1]))
            if violations.any():
                errors.append(f"{baseline_col}: {violations.sum()} values outside bounds {bounds}")
    
    # Check final values (baseline + delta) are within bounds
    for col in [c for c in df.columns if c.endswith('_baseline')]:
        lab = col.replace('_baseline', '')
        delta_col = f'{lab}_delta'
        if delta_col in df.columns:
            # Calculate final values
            baseline = df[col]
            delta = df[delta_col]
            final_values = baseline + delta
            
            # Check bounds (only where both baseline and delta are not missing)
            both_present = baseline.notna() & delta.notna()
            bounds = LAB_BOUNDS.get(lab, (0, np.inf))
            violations = both_present & ((final_values < bounds[0]) | (final_values > bounds[1]))
            if violations.any():
                errors.append(f"{lab} final values: {violations.sum()} outside bounds {bounds} (baseline + delta)")
    
    # Check for negative baseline values (baselines should never be negative)
    for col in [c for c in df.columns if c.endswith('_baseline')]:
        if df[col].notna().any():
            negatives = (df[col] < 0).sum()
            if negatives > 0:
                errors.append(f"{col}: {negatives} negative baseline values")
    
    return len(errors) == 0, errors


def run_comprehensive_validation(df: pd.DataFrame, ground_truth: Dict) -> Dict:
    """
    Comprehensive validation test suite.
    
    Returns detailed report of all checks.
    """
    validation_report = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(df),
        'checks': []
    }
    
    # CHECK 1: Baseline values within physiological bounds (deltas are changes, not checked)
    for lab, bounds in LAB_BOUNDS.items():
        baseline_col = f'{lab}_baseline'
        if baseline_col in df.columns:
            violations = df[baseline_col].notna() & ((df[baseline_col] < bounds[0]) | (df[baseline_col] > bounds[1]))
            validation_report['checks'].append({
                'check': f'{baseline_col}_bounds',
                'passed': bool(violations.sum() == 0),  # Convert numpy bool to Python bool
                'violations': int(violations.sum()),
                'details': f'Expected: {bounds}, Got: [{float(df[baseline_col].min()):.2f}, {float(df[baseline_col].max()):.2f}]'
            })
    
    # CHECK 2: Drug effects have correct sign
    for drug in df['drug_name'].unique():
        drug_data = df[df['drug_name'] == drug]
        expected_effects = {k: v for k, v in DRUG_EFFECTS.get(drug, {}).items() 
                          if k != 'tau_days'}
        
        for lab, expected_sign in expected_effects.items():
            delta_col = f'{lab}_delta'
            if delta_col in drug_data.columns and drug_data[delta_col].notna().any():
                observed_mean = float(drug_data[delta_col].mean())  # Convert to Python float
                correct_sign = bool(np.sign(observed_mean) == np.sign(expected_sign))  # Convert to Python bool
                
                validation_report['checks'].append({
                    'check': f'{drug}_{lab}_sign',
                    'passed': correct_sign,
                    'expected': f'{"decrease" if expected_sign < 0 else "increase"}',
                    'observed': f'{observed_mean:.2f}'
                })
    
    # CHECK 3: Baseline + delta stays within bounds
    for col in [c for c in df.columns if c.endswith('_baseline')]:
        lab = col.replace('_baseline', '')
        delta_col = f'{lab}_delta'
        
        if delta_col in df.columns:
            final_values = df[col] + df[delta_col]
            bounds = LAB_BOUNDS.get(lab, (0, np.inf))
            violations = final_values.notna() & ((final_values < bounds[0]) | (final_values > bounds[1]))
            
            validation_report['checks'].append({
                'check': f'{lab}_final_values_in_bounds',
                'passed': bool(violations.sum() == 0),  # Convert numpy bool to Python bool
                'violations': int(violations.sum())
            })
    
    # CHECK 4: Statistical properties match expectations
    for drug in df['drug_name'].unique():
        drug_data = df[df['drug_name'] == drug]
        expected_effects = {k: v for k, v in DRUG_EFFECTS.get(drug, {}).items() 
                          if k != 'tau_days'}
        
        for lab, expected_effect in expected_effects.items():
            delta_col = f'{lab}_delta'
            if delta_col in drug_data.columns and drug_data[delta_col].notna().sum() > 0:
                observed_mean = float(drug_data[delta_col].mean())  # Convert to Python float
                # Allow 30% deviation from expected
                within_tolerance = bool(abs(observed_mean - expected_effect) < abs(expected_effect) * 0.3)  # Convert to Python bool
                
                deviation_pct = float(abs((observed_mean - expected_effect) / expected_effect * 100)) if expected_effect != 0 else 0.0
                
                validation_report['checks'].append({
                    'check': f'{drug}_{lab}_magnitude',
                    'passed': within_tolerance,
                    'expected': float(expected_effect),  # Ensure float
                    'observed': observed_mean,
                    'deviation_pct': deviation_pct
                })
    
    # Summary
    total_checks = len(validation_report['checks'])
    passed_checks = sum(1 for c in validation_report['checks'] if c['passed'])
    validation_report['summary'] = {
        'total_checks': int(total_checks),
        'passed': int(passed_checks),
        'failed': int(total_checks - passed_checks),
        'pass_rate': float(passed_checks / total_checks * 100) if total_checks > 0 else 0.0
    }
    
    return validation_report


# ============================================================================
# GROUND TRUTH & METADATA
# ============================================================================

def save_ground_truth_metadata():
    """Save the true data generating process for benchmarking."""
    ground_truth = {
        'version': '1.0',
        'generated_date': datetime.now().isoformat(),
        'true_causal_effects': {},
        'confounding_structure': {
            'age': {
                'affects': ['LBXSGL', 'LBXTC', 'LBDLDL', 'BPXSY1'],
                'relationship': 'linear_positive',
                'strength': 'moderate'
            },
            'bmi': {
                'affects': ['LBXSGL', 'LBXTC', 'BPXSY1'],
                'relationship': 'nonlinear_positive',
                'strength': 'strong'
            },
            'sex': {
                'affects': ['LBDSCR', 'LBXSCH'],
                'relationship': 'categorical',
                'strength': 'moderate'
            }
        },
        'noise_model': {
            'individual_variability': {'type': 'normal', 'cv': CONFIG['individual_cv']},
            'measurement_error': {'type': 'normal', 'cv': CONFIG['measurement_error_cv']},
        },
        'data_quality': {
            'missing_mechanism': 'MAR',
            'missingness_rate': CONFIG['missing_rate'],
            'outlier_rate': CONFIG['outlier_rate']
        }
    }
    
    # Add drug effects
    for drug, effects in DRUG_EFFECTS.items():
        if drug == 'tau_days':
            continue
        ground_truth['true_causal_effects'][drug] = {}
        for lab, effect in effects.items():
            if lab != 'tau_days':
                ground_truth['true_causal_effects'][drug][lab] = {
                    'mean_effect': effect,
                    'effect_type': 'additive',
                    'modifiers': ['adherence', 'time', 'individual_variability'],
                    'confounders': ['age', 'bmi', 'sex'],
                    'tau_days': DRUG_EFFECTS[drug].get('tau_days', 30)
                }
    
    output_path = Path('data/cdisc/ground_truth.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    return ground_truth


def calculate_expected_benchmarks(responses_df: pd.DataFrame) -> Dict:
    """Calculate what a perfect model should recover."""
    benchmarks = {}
    
    for drug in responses_df['drug_name'].unique():
        drug_data = responses_df[responses_df['drug_name'] == drug]
        benchmarks[drug] = {}
        
        for col in [c for c in drug_data.columns if c.endswith('_delta')]:
            lab = col.replace('_delta', '')
            if drug_data[col].notna().sum() > 0:
                benchmarks[drug][lab] = {
                    'true_mean_effect': DRUG_EFFECTS[drug].get(lab, 0),
                    'observed_mean': float(drug_data[col].mean()),
                    'observed_std': float(drug_data[col].std()),
                    'sample_size': int(drug_data[col].notna().sum()),
                    'effect_size_cohen_d': float(drug_data[col].mean() / drug_data[col].std()) if drug_data[col].std() > 0 else 0
                }
    
    output_path = Path('data/cdisc/expected_benchmarks.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(benchmarks, f, indent=2)
    
    return benchmarks


# ============================================================================
# DATASET VARIANTS
# ============================================================================

def generate_dataset_variants(base_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Generate multiple dataset variants with different characteristics."""
    
    variants = {
        'perfect_clean': base_df.copy(),  # No noise, no missing data
        
        'realistic': add_realistic_data_quality_issues(
            base_df.copy(),
            missing_rate=CONFIG['missing_rate'],
            measurement_error_cv=CONFIG['measurement_error_cv']
        ),
        
        'high_noise': add_realistic_data_quality_issues(
            base_df.copy(),
            missing_rate=0.15,
            measurement_error_cv=0.08
        ),
        
        'small_sample': base_df.sample(n=500, random_state=42).reset_index(drop=True),
    }
    
    # Save each variant
    output_dir = Path('data/cdisc')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, df in variants.items():
        output_path = output_dir / f'responses_{name}.csv'
        df.to_csv(output_path, index=False)
        missing_count = df.isnull().sum().sum()
        print(f"  ✓ Saved {name}: {len(df)} samples, {missing_count} missing values")
    
    return variants


# ============================================================================
# DATA DICTIONARY
# ============================================================================

def generate_data_dictionary():
    """Create comprehensive data dictionary with metadata."""
    
    data_dict = []
    
    # Demographics
    data_dict.extend([
        {
            'variable': 'patient_id',
            'type': 'integer',
            'description': 'Unique patient identifier',
            'range': f'0-{CONFIG["n_patients"]-1}',
            'missing_allowed': False
        },
        {
            'variable': 'age',
            'type': 'integer',
            'description': 'Patient age at treatment start',
            'unit': 'years',
            'range': '18-90',
            'distribution': 'Normal(μ=55, σ=15)',
            'missing_allowed': False
        },
        {
            'variable': 'sex',
            'type': 'categorical',
            'description': 'Patient sex',
            'values': ['M', 'F'],
            'missing_allowed': False
        },
        {
            'variable': 'bmi',
            'type': 'float',
            'description': 'Body Mass Index',
            'unit': 'kg/m²',
            'range': '15-50',
            'missing_allowed': False
        },
        {
            'variable': 'days_on_drug',
            'type': 'integer',
            'description': 'Number of days patient has been on drug',
            'unit': 'days',
            'range': '1-180',
            'missing_allowed': False
        },
        {
            'variable': 'adherence',
            'type': 'float',
            'description': 'Medication adherence rate',
            'unit': 'proportion',
            'range': f'{CONFIG["min_adherence"]}-{CONFIG["max_adherence"]}',
            'missing_allowed': False
        },
    ])
    
    # Adverse events
    for event in ['statin_myalgia', 'ace_cough', 'metformin_gi_upset']:
        data_dict.append({
            'variable': event,
            'type': 'binary',
            'description': f'Presence of {event}',
            'values': [0, 1],
            'missing_allowed': False
        })
    
    # Lab values
    for lab, bounds in LAB_BOUNDS.items():
        data_dict.extend([
            {
                'variable': f'{lab}_baseline',
                'type': 'float',
                'description': f'{lab} value before treatment',
                'range': f'{bounds[0]}-{bounds[1]}',
                'missing_allowed': True,
                'missing_mechanism': 'MAR'
            },
            {
                'variable': f'{lab}_delta',
                'type': 'float',
                'description': f'Change in {lab} caused by drug',
                'missing_allowed': True
            }
        ])
    
    # Save as CSV
    output_path = Path('data/cdisc/data_dictionary.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data_dict).to_csv(output_path, index=False)
    
    return data_dict


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function."""
    # Set random seed ONCE at the start
    np.random.seed(42)
    
    print("=" * 70)
    print("GENERATING PERFECT SYNTHETIC DRUG-RESPONSE DATASET")
    print("=" * 70)
    print("\nBased on published clinical trial data and FDA drug labels")
    print("With realistic demographics, temporal dynamics, and adherence\n")
    
    # Generate patients
    print("[1/8] Generating patient demographics...")
    patients_df = generate_realistic_patient_demographics()
    print(f"  ✅ Generated {len(patients_df)} patients")
    
    # Generate baseline labs
    print("\n[2/8] Generating baseline lab values with demographic dependencies...")
    baseline_labs_df = generate_baseline_labs(patients_df)
    print(f"  ✅ Generated {baseline_labs_df.shape[1]} baseline biomarkers")
    
    # Generate drug responses
    print("\n[3/8] Generating drug-response pairs...")
    responses_df = generate_drug_responses(patients_df, baseline_labs_df)
    print(f"  ✅ Generated {len(responses_df)} drug-response samples")
    
    # Validate BEFORE saving
    print("\n[4/8] Validating generated data...")
    is_valid, errors = validate_dataset(responses_df)
    if not is_valid:
        print(f"  ❌ Validation failed with {len(errors)} errors:")
        for error in errors[:10]:
            print(f"    - {error}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more errors")
        raise ValueError("Data validation failed - not saving corrupted data")
    print(f"  ✅ All validation checks passed")
    
    # Save clean version
    output_path = Path('data/cdisc/clinical_trials_responses.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    responses_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved clean data to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Save ground truth metadata
    print("\n[5/8] Saving ground truth metadata...")
    save_ground_truth_metadata()
    print(f"  ✅ Saved ground truth to: data/cdisc/ground_truth.json")
    
    # Calculate expected benchmarks
    print("\n[6/8] Calculating expected benchmarks...")
    benchmarks = calculate_expected_benchmarks(responses_df)
    print(f"  ✅ Saved benchmarks to: data/cdisc/expected_benchmarks.json")
    
    # Generate dataset variants
    print("\n[7/8] Generating dataset variants...")
    variants = generate_dataset_variants(responses_df)
    
    # Generate data dictionary
    print("\n[8/8] Generating data dictionary...")
    data_dict = generate_data_dictionary()
    print(f"  ✅ Saved data dictionary to: data/cdisc/data_dictionary.csv")
    
    # Run comprehensive validation
    print("\n[BONUS] Running comprehensive validation...")
    validation_report = run_comprehensive_validation(responses_df, DRUG_EFFECTS)
    
    # Convert all numpy types to native Python types before JSON serialization
    def convert_to_native_types(obj):
        """Recursively convert numpy types to native Python types (NumPy 2.0 compatible)."""
        # Check for numpy scalar types (NumPy 2.0 compatible)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native_types(item) for item in obj]
        # Also handle pandas/numpy dtypes that might be in the data
        elif hasattr(obj, 'dtype') and isinstance(obj.dtype, (np.integer, np.floating, np.bool_)):
            return convert_to_native_types(obj.item() if hasattr(obj, 'item') else obj)
        return obj
    
    # Convert validation report
    validation_report = convert_to_native_types(validation_report)
    
    # Save validation report
    validation_path = Path('data/cdisc/validation_report.json')
    with open(validation_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ PERFECT SYNTHETIC DATASET GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  📊 clinical_trials_responses.csv - Clean training data")
    print(f"  📊 responses_perfect_clean.csv - Perfect variant")
    print(f"  📊 responses_realistic.csv - Realistic with noise/missing")
    print(f"  📊 responses_high_noise.csv - Challenging variant")
    print(f"  📊 responses_small_sample.csv - Limited data variant")
    print(f"  📋 ground_truth.json - True causal effects")
    print(f"  📋 expected_benchmarks.json - What models should recover")
    print(f"  📋 validation_report.json - Data quality report")
    print(f"  📖 data_dictionary.csv - Variable documentation")
    
    print(f"\n🎯 Your synthetic data is now PERFECT for:")
    print(f"  • Training ML models")
    print(f"  • Benchmarking causal inference methods")
    print(f"  • Testing with known ground truth")
    print(f"  • Evaluating model robustness")
    
    # Summary statistics
    print(f"\n📊 Dataset Summary:")
    print(f"  Total samples: {len(responses_df):,}")
    print(f"  Unique patients: {responses_df['patient_id'].nunique():,}")
    print(f"  Unique drugs: {responses_df['drug_name'].nunique()}")
    print(f"  Validation: {validation_report['summary']['passed']}/{validation_report['summary']['total_checks']} checks passed ({validation_report['summary']['pass_rate']:.1f}%)")


if __name__ == '__main__':
    main()
