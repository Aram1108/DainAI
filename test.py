"""
IV Drug Pharmacokinetics - Synthetic Data Generator

Generates high-frequency (5-second intervals) patient data for IV drug testing.
Models drug concentration using PK models and lab value changes using Emax models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import Dict, List, Tuple
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.lab_reference_ranges import LAB_RANGES, CRITICAL_RULES, get_sex_adjusted_range
from models.patient_generator_gan import PatientGenerator


# ============================================================================
# PHARMACOKINETIC MODELS
# ============================================================================

def drug_concentration(time_sec: float, dose_mg: float, volume_L: float, half_life_min: float) -> float:
    """
    Calculate drug concentration at given time after IV bolus (one-compartment model).
    
    Args:
        time_sec: Time since injection (seconds)
        dose_mg: Initial dose (mg)
        volume_L: Volume of distribution (liters)
        half_life_min: Drug half-life (minutes)
    
    Returns:
        Concentration in mg/L
    """
    if time_sec < 0:
        return 0.0
    
    t_min = time_sec / 60.0
    k_elim = 0.693 / half_life_min  # Elimination rate constant
    C0 = dose_mg / volume_L  # Initial concentration
    C_t = C0 * np.exp(-k_elim * t_min)
    
    return max(0.0, C_t)  # Never negative


def calculate_effect(drug_conc: float, EC50: float, Emax: float, baseline: float, 
                     effect_direction: str = 'decrease') -> float:
    """
    Calculate drug effect on lab value using Emax model.
    
    Args:
        drug_conc: Current drug concentration (mg/L)
        EC50: Concentration for 50% max effect (mg/L)
        Emax: Maximum effect (units of lab value)
        baseline: Baseline lab value
        effect_direction: 'decrease' or 'increase'
    
    Returns:
        New lab value
    """
    if drug_conc <= 0:
        return baseline
    
    effect = (Emax * drug_conc) / (EC50 + drug_conc)
    
    if effect_direction == 'decrease':
        return baseline - effect
    else:
        return baseline + effect


def add_measurement_noise(value: float, cv_percent: float = 3.0) -> float:
    """Add realistic measurement variability."""
    noise = np.random.normal(0, cv_percent / 100.0)
    return value * (1.0 + noise)


def add_biological_variability(value: float, time_sec: float, cv_percent: float = 5.0) -> float:
    """Add biological fluctuation (heartbeat, breathing, circadian rhythm)."""
    # Use sine wave for periodic variation (circadian + short-term)
    time_min = time_sec / 60.0
    circadian = np.sin(2 * np.pi * time_min / (24 * 60)) * (cv_percent / 200.0)  # 24-hour cycle
    short_term = np.sin(2 * np.pi * time_min / 5.0) * (cv_percent / 400.0)  # 5-minute cycle
    random_component = np.random.normal(0, cv_percent / 200.0)
    
    return value * (1.0 + circadian + short_term + random_component)


# ============================================================================
# DRUG CONFIGURATIONS
# ============================================================================

DRUG_CONFIGS = {
    'Regular Insulin': {
        'name': 'Regular Insulin',
        'dose_mg': 10.0,
        'half_life_min': 5.0,
        'volume_L': 15.0,
        'injection_time_sec': 0,
        'primary_effect': {
            'lab': 'LBXSGL',
            'EC50': 0.5,
            'Emax': 85.0,
            'direction': 'decrease'
        },
        'secondary_effects': {
            'LBXSK': {
                'EC50': 0.5,
                'Emax': 0.5,
                'direction': 'decrease'
            }
        }
    },
    'Furosemide': {
        'name': 'Furosemide',
        'dose_mg': 40.0,
        'half_life_min': 75.0,  # 60-90 min average
        'volume_L': 15.0,
        'injection_time_sec': 0,
        'primary_effect': {
            'lab': 'LBXSK',
            'EC50': 2.0,
            'Emax': 0.6,
            'direction': 'decrease'
        },
        'secondary_effects': {
            'LBXSNA': {
                'EC50': 2.0,
                'Emax': 3.0,
                'direction': 'decrease'
            },
            'LBXSCA': {
                'EC50': 2.0,
                'Emax': 0.3,
                'direction': 'increase'
            },
            'LBDSCR': {
                'EC50': 2.0,
                'Emax': 0.2,
                'direction': 'increase'
            },
            'LBXSBU': {
                'EC50': 2.0,
                'Emax': 4.0,
                'direction': 'increase'
            }
        }
    },
    'Potassium Chloride': {
        'name': 'Potassium Chloride',
        'dose_mg': 1560.0,  # 20 mEq = 1560 mg
        'half_life_min': 120.0,  # Slow elimination
        'volume_L': 40.0,  # ECF volume
        'injection_time_sec': 0,
        'infusion_duration_sec': 3600,  # 1 hour infusion
        'primary_effect': {
            'lab': 'LBXSK',
            'EC50': 10.0,
            'Emax': 1.5,
            'direction': 'increase'
        },
        'secondary_effects': {}
    },
    'Epinephrine': {
        'name': 'Epinephrine',
        'dose_mg': 0.5,
        'half_life_min': 2.5,
        'volume_L': 60.0,
        'injection_time_sec': 0,
        'primary_effect': {
            'lab': 'LBXSGL',
            'EC50': 0.01,
            'Emax': 30.0,
            'direction': 'increase'
        },
        'secondary_effects': {
            'LBXSK': {
                'EC50': 0.01,
                'Emax': 0.3,
                'direction': 'increase'
            }
        }
    }
}


# ============================================================================
# BASELINE LAB GENERATION
# ============================================================================

def generate_baseline_labs(age: int, sex: str, drug_name: str) -> Dict[str, float]:
    """
    Generate realistic baseline lab values adjusted for patient demographics.
    
    For insulin testing, start with hyperglycemia.
    """
    labs = {}
    
    # Get sex-adjusted ranges
    sex_code = sex
    
    # Fast-changing labs (will change during experiment)
    if drug_name == 'Regular Insulin':
        # Start hyperglycemic for insulin test
        labs['LBXSGL'] = np.random.uniform(160, 200)
    else:
        labs['LBXSGL'] = np.random.uniform(85, 110)
    
    labs['LBXSK'] = np.random.uniform(3.8, 4.5)
    labs['LBXSNA'] = np.random.uniform(138, 142)
    labs['LBXSCA'] = np.random.uniform(9.0, 9.8)
    labs['LBXSOS'] = np.random.uniform(280, 290)
    
    # Slow-changing labs (essentially constant over 24hr)
    labs['LBXTC'] = np.random.uniform(160, 220)
    labs['LBXSCL'] = np.random.uniform(90, 140)
    
    # HDL: sex-specific
    if sex == 'F':
        labs['LBDSCH'] = np.random.uniform(50, 70)
    else:
        labs['LBDSCH'] = np.random.uniform(40, 60)
    
    labs['LBDTC'] = np.random.uniform(80, 180)
    
    # Liver function (constant)
    labs['LBXSGT'] = np.random.uniform(15, 40)
    labs['LBXSAS'] = np.random.uniform(15, 35)
    labs['LBXSGB'] = np.random.uniform(0.3, 1.0)
    labs['LBXSAL'] = np.random.uniform(3.8, 4.5)
    labs['LBXSTP'] = np.random.uniform(6.5, 7.8)
    
    # Kidney function (slow change)
    if sex == 'M':
        labs['LBDSCR'] = np.random.uniform(0.8, 1.2)
        labs['LBXSUA'] = np.random.uniform(4.5, 6.5)
    else:
        labs['LBDSCR'] = np.random.uniform(0.6, 1.0)
        labs['LBXSUA'] = np.random.uniform(3.5, 5.5)
    
    labs['LBXSBU'] = np.random.uniform(10, 18)
    
    # Other (constant)
    labs['LBXCRP'] = np.random.uniform(0.5, 2.0)
    labs['LBXBCD'] = np.random.uniform(0.2, 1.0)
    labs['LBXBPB'] = np.random.uniform(0.8, 3.0)
    
    # Clamp to physiological limits
    for lab_name, value in labs.items():
        if lab_name in LAB_RANGES and LAB_RANGES[lab_name]['physiological_limit']:
            min_phys, max_phys = LAB_RANGES[lab_name]['physiological_limit']
            labs[lab_name] = np.clip(value, min_phys, max_phys)
    
    return labs


# ============================================================================
# TIME SERIES GENERATION
# ============================================================================

def generate_timeseries(
    patient_id: int,
    patient: Dict,
    baseline_labs: Dict[str, float],
    drug_config: Dict,
    duration_sec: int = 86400,
    interval_sec: int = 5
) -> pd.DataFrame:
    """
    Generate complete time series for one patient on one drug.
    
    Returns DataFrame with 17,280 rows (one per 5 seconds for 24 hours).
    """
    times = np.arange(0, duration_sec + interval_sec, interval_sec)
    data_rows = []
    
    injection_time = drug_config.get('injection_time_sec', 0)
    infusion_duration = drug_config.get('infusion_duration_sec', 0)
    
    for t in times:
        row = {
            'patient_id': patient_id,
            'time_sec': int(t),
            'age': patient['age'],
            'sex': patient['sex'],
            'height_cm': patient['height_cm'],
            'weight_kg': patient['weight_kg'],
            'bmi': round(patient['bmi'], 1),
            'drug_name': drug_config['name'],
            'drug_dose_mg': drug_config['dose_mg'],
        }
        
        # Calculate drug concentration at time t
        if t >= injection_time:
            t_since_injection = t - injection_time
            
            # Handle infusion vs bolus
            if infusion_duration > 0 and t_since_injection < infusion_duration:
                # During infusion: concentration rises
                # Simplified: linear rise during infusion
                infusion_conc = drug_concentration(
                    infusion_duration,
                    drug_config['dose_mg'],
                    drug_config['volume_L'],
                    drug_config['half_life_min']
                )
                drug_conc = infusion_conc * (t_since_injection / infusion_duration)
            else:
                # After infusion or bolus: exponential decay
                drug_conc = drug_concentration(
                    t_since_injection,
                    drug_config['dose_mg'],
                    drug_config['volume_L'],
                    drug_config['half_life_min']
                )
        else:
            drug_conc = 0.0
        
        row['drug_concentration_mg_L'] = round(drug_conc, 2)
        
        # Calculate primary effect
        primary = drug_config['primary_effect']
        baseline_value = baseline_labs[primary['lab']]
        
        effect_value = calculate_effect(
            drug_conc,
            primary['EC50'],
            primary['Emax'],
            baseline_value,
            primary['direction']
        )
        
        # Add noise
        noisy_value = add_measurement_noise(
            add_biological_variability(effect_value, t, cv_percent=5.0),
            cv_percent=3.0
        )
        
        # Clamp to physiological limits
        if primary['lab'] in LAB_RANGES and LAB_RANGES[primary['lab']]['physiological_limit']:
            min_phys, max_phys = LAB_RANGES[primary['lab']]['physiological_limit']
            noisy_value = np.clip(noisy_value, min_phys, max_phys)
        
        row[primary['lab']] = round(noisy_value, 1)
        
        # Calculate secondary effects
        for lab_name, effect_params in drug_config.get('secondary_effects', {}).items():
            baseline_value = baseline_labs[lab_name]
            
            effect_value = calculate_effect(
                drug_conc,
                effect_params['EC50'],
                effect_params['Emax'],
                baseline_value,
                effect_params['direction']
            )
            
            noisy_value = add_measurement_noise(
                add_biological_variability(effect_value, t, cv_percent=3.0),
                cv_percent=2.0
            )
            
            # Clamp to physiological limits
            if lab_name in LAB_RANGES and LAB_RANGES[lab_name]['physiological_limit']:
                min_phys, max_phys = LAB_RANGES[lab_name]['physiological_limit']
                noisy_value = np.clip(noisy_value, min_phys, max_phys)
            
            row[lab_name] = round(noisy_value, 1)
        
        # All other labs: add small random walk or keep constant
        for lab_name, baseline_value in baseline_labs.items():
            if lab_name not in row:
                # Slow-changing labs: add tiny drift
                if lab_name in ['LBXTC', 'LBXSCL', 'LBDSCH', 'LBDTC', 'LBXSGT', 'LBXSAS', 
                               'LBXBCD', 'LBXBPB']:
                    noisy_value = baseline_value * (1.0 + np.random.normal(0, 0.001))
                # Electrolytes: very stable with noise
                elif lab_name in ['LBXSNA', 'LBXSCA', 'LBXSOS']:
                    noisy_value = baseline_value * (1.0 + np.random.normal(0, 0.002))
                # Others: constant with measurement noise
                else:
                    noisy_value = baseline_value * (1.0 + np.random.normal(0, 0.005))
                
                # Clamp to physiological limits
                if lab_name in LAB_RANGES and LAB_RANGES[lab_name]['physiological_limit']:
                    min_phys, max_phys = LAB_RANGES[lab_name]['physiological_limit']
                    noisy_value = np.clip(noisy_value, min_phys, max_phys)
                
                # Round based on lab type
                if lab_name in ['LBDSCR']:
                    row[lab_name] = round(noisy_value, 2)
                elif lab_name in ['LBXSGL', 'LBXSK', 'LBXSNA', 'LBXSCA', 'LBXSOS', 
                                 'LBXSGB', 'LBXSAL', 'LBXSTP', 'LBXSUA', 'LBXCRP', 
                                 'LBXBCD', 'LBXBPB']:
                    row[lab_name] = round(noisy_value, 1)
                else:
                    row[lab_name] = round(noisy_value, 0)
        
        data_rows.append(row)
    
    return pd.DataFrame(data_rows)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_data(df: pd.DataFrame) -> List[str]:
    """Check for biological impossibilities."""
    errors = []
    
    # Check for negative values in labs that must be positive
    positive_labs = ['LBXSGL', 'LBXSK', 'LBXSNA', 'LBXSGT', 'LBXCRP', 'LBXSAS', 
                     'LBXSGB', 'LBXSAL', 'LBXSTP', 'LBXTC', 'LBDSCH', 'LBXSCL', 
                     'LBDTC', 'LBXBCD', 'LBXBPB']
    
    for lab in positive_labs:
        if lab in df.columns and (df[lab] <= 0).any():
            errors.append(f"{lab} has non-positive values!")
    
    # Check for dangerous electrolyte levels
    if 'LBXSK' in df.columns:
        if (df['LBXSK'] < 2.5).any() or (df['LBXSK'] > 6.0).any():
            errors.append("Potassium outside safe range (2.5-6.0)!")
    
    if 'LBXSNA' in df.columns:
        if (df['LBXSNA'] < 125).any() or (df['LBXSNA'] > 155).any():
            errors.append("Sodium outside safe range (125-155)!")
    
    # Check glucose doesn't drop too low (hypoglycemia)
    if 'LBXSGL' in df.columns:
        if (df['LBXSGL'] < 50).any():
            errors.append("Dangerous hypoglycemia (<50 mg/dL)!")
    
    return errors


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_patient_data(
    patient_id: int,
    age: int,
    sex: str,
    height_cm: float,
    weight_kg: float,
    drug_name: str,
    output_dir: Path
) -> Path:
    """
    Generate complete time series data for one patient on one drug.
    
    Returns path to saved CSV file.
    """
    # Create patient directory
    patient_dir = output_dir / f"patient_{patient_id:03d}"
    patient_dir.mkdir(parents=True, exist_ok=True)
    
    # Patient info
    patient = {
        'age': age,
        'sex': sex,
        'height_cm': height_cm,
        'weight_kg': weight_kg,
        'bmi': weight_kg / (height_cm / 100) ** 2
    }
    
    # Generate baseline labs
    baseline_labs = generate_baseline_labs(age, sex, drug_name)
    
    # Get drug config
    if drug_name not in DRUG_CONFIGS:
        raise ValueError(f"Unknown drug: {drug_name}. Available: {list(DRUG_CONFIGS.keys())}")
    
    drug_config = DRUG_CONFIGS[drug_name]
    
    # Generate time series
    print(f"  Generating time series for patient {patient_id} on {drug_name}...")
    df = generate_timeseries(
        patient_id=patient_id,
        patient=patient,
        baseline_labs=baseline_labs,
        drug_config=drug_config,
        duration_sec=86400,  # 24 hours
        interval_sec=30  # Every 30 seconds
    )
    
    # Validate
    errors = validate_data(df)
    if errors:
        print(f"  ⚠ Warnings for patient {patient_id}:")
        for error in errors:
            print(f"    - {error}")
    else:
        print(f"  ✓ Patient {patient_id} data validated")
    
    # Save
    filename = f"patient_{patient_id:03d}_{drug_name.replace(' ', '_').lower()}_24hr.csv"
    filepath = patient_dir / filename
    df.to_csv(filepath, index=False)
    
    print(f"  ✓ Saved: {filepath} ({len(df):,} rows)")
    
    return filepath


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    """Generate test data for 10 patients on multiple drugs."""
    print("=" * 70)
    print("IV DRUG PHARMACOKINETICS - SYNTHETIC DATA GENERATOR")
    print("=" * 70)
    
    # Create timestamped output directory
    timestamp = time.strftime("%Y-%m-%d_%Hh%Mm")
    results_dir = Path('results') / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {results_dir}")
    
    # Generate 10 patients with different demographics
    patients = [
        {'age': 25, 'sex': 'M', 'height_cm': 180, 'weight_kg': 75},
        {'age': 35, 'sex': 'F', 'height_cm': 165, 'weight_kg': 60},
        {'age': 45, 'sex': 'M', 'height_cm': 175, 'weight_kg': 85},
        {'age': 55, 'sex': 'F', 'height_cm': 160, 'weight_kg': 70},
        {'age': 65, 'sex': 'M', 'height_cm': 170, 'weight_kg': 80},
        {'age': 30, 'sex': 'F', 'height_cm': 168, 'weight_kg': 65},
        {'age': 40, 'sex': 'M', 'height_cm': 178, 'weight_kg': 90},
        {'age': 50, 'sex': 'F', 'height_cm': 162, 'weight_kg': 68},
        {'age': 28, 'sex': 'M', 'height_cm': 182, 'weight_kg': 78},
        {'age': 60, 'sex': 'F', 'height_cm': 158, 'weight_kg': 72},
    ]
    
    # Drugs to test
    drugs_to_test = ['Regular Insulin', 'Furosemide', 'Potassium Chloride', 'Epinephrine']
    
    print(f"\nGenerating data for {len(patients)} patients on {len(drugs_to_test)} drugs...")
    print(f"Total simulations: {len(patients) * len(drugs_to_test)}")
    print(f"Rows per simulation: 17,280 (24 hours × 5-second intervals)")
    print(f"Total rows: {len(patients) * len(drugs_to_test) * 17280:,}\n")
    
    generated_files = []
    
    for patient_id, patient in enumerate(patients, start=1):
        print(f"\n[Patient {patient_id}/{len(patients)}]")
        print(f"  Demographics: {patient['age']}yo {patient['sex']}, "
              f"{patient['height_cm']}cm, {patient['weight_kg']}kg")
        
        for drug_name in drugs_to_test:
            try:
                filepath = generate_patient_data(
                    patient_id=patient_id,
                    age=patient['age'],
                    sex=patient['sex'],
                    height_cm=patient['height_cm'],
                    weight_kg=patient['weight_kg'],
                    drug_name=drug_name,
                    output_dir=results_dir
                )
                generated_files.append(filepath)
            except Exception as e:
                print(f"  ✗ Error generating {drug_name} for patient {patient_id}: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Generated {len(generated_files)} files")
    print(f"Output directory: {results_dir}")
    print(f"\nFolder structure:")
    print(f"  {results_dir.name}/")
    for patient_id in range(1, len(patients) + 1):
        patient_dir = results_dir / f"patient_{patient_id:03d}"
        if patient_dir.exists():
            files = list(patient_dir.glob("*.csv"))
            print(f"    patient_{patient_id:03d}/ ({len(files)} files)")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()

