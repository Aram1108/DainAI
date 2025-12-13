"""
Extract and augment real CDISC clinical trial data for training.

This module:
1. Extracts real patient-drug-response data from CDISC files (adsl.csv + adlbc.csv)
2. Augments the real data with variations (noise, scaling, etc.)
3. Returns ONLY augmentations (original data excluded from training)
4. Balances augmented data to match synthetic data size
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Mapping from CDISC parameter codes to our lab biomarker features
CDISC_TO_LAB_MAP = {
    'SODIUM': 'LBXSNA',
    'K': 'LBXSK',
    'CL': 'LBXSCL',
    'BILI': 'LBXSGB',
    'CREAT': 'LBDSCR',
    'UREAN': 'LBXSBU',
    'CHOL': 'LBXTC',
    'GLUC': 'LBXSGL',
    'AST': 'LBXSAS',
    'ALT': 'LBXSAL',
    'ALP': 'LBXSGT',
    'CA': 'LBXSCA',
    'PHOS': 'LBXSTP',
    'URIC': 'LBXSUA',
    'HDL': 'LBXSCH',
    'LDL': 'LBDLDL',
    'TRIG': 'LBDTC',
    'CRP': 'LBXCRP',
}

# Reverse mapping
LAB_TO_CDISC_MAP = {v: k for k, v in CDISC_TO_LAB_MAP.items()}


def extract_real_cdisc_data(
    adsl_path: str = 'data/cdisc/adsl.csv',
    adlbc_path: str = 'data/cdisc/adlbc.csv',
    drug_smiles_path: str = 'data/mappings/drug_smiles_mapping.json'
) -> pd.DataFrame:
    """
    Extract real patient-drug-response data from CDISC files.
    
    Args:
        adsl_path: Path to adsl.csv (subject-level data)
        adlbc_path: Path to adlbc.csv (lab data)
        drug_smiles_path: Path to drug SMILES mapping
        
    Returns:
        DataFrame with columns: patient_id, drug_name, age, sex, bmi, 
        and lab delta/baseline columns matching clinical_trials_responses.csv format
    """
    print("\n" + "="*80)
    print("EXTRACTING REAL CDISC DATA")
    print("="*80)
    
    # Load subject-level data
    print(f"\n[1/4] Loading subject data from {adsl_path}...")
    adsl = pd.read_csv(adsl_path)
    print(f"  ✓ Loaded {len(adsl):,} subjects")
    
    # Load lab data
    print(f"\n[2/4] Loading lab data from {adlbc_path}...")
    adlbc = pd.read_csv(adlbc_path)
    print(f"  ✓ Loaded {len(adlbc):,} lab records")
    
    # Load drug SMILES mapping
    print(f"\n[3/4] Loading drug SMILES mapping...")
    drug_smiles = {}
    drug_name_mapping = {}  # Maps base names to full names with SMILES
    if Path(drug_smiles_path).exists():
        import json
        with open(drug_smiles_path, 'r') as f:
            drug_smiles = json.load(f)
        print(f"  ✓ Loaded {len(drug_smiles)} drug mappings")
        
        # Create reverse mapping: base name -> full name with SMILES
        # Handle special cases like "Xanomeline" -> "Xanomeline Low Dose"
        for full_name, smiles in drug_smiles.items():
            if smiles is not None:  # Only map drugs with valid SMILES
                # Extract base name (first word)
                base_name = full_name.split()[0] if ' ' in full_name else full_name
                # Prefer shorter names (e.g., "Low Dose" over "High Dose")
                if base_name not in drug_name_mapping or len(full_name) < len(drug_name_mapping[base_name]):
                    drug_name_mapping[base_name] = full_name
    else:
        print(f"  ⚠️  Drug SMILES mapping not found, will use drug names as-is")
    
    # Extract baseline and post-treatment lab values
    print(f"\n[4/4] Processing lab changes...")
    
    # Filter to baseline and post-baseline visits
    adlbc_baseline = adlbc[adlbc['AVISIT'].str.contains('Baseline', case=False, na=False)].copy()
    adlbc_post = adlbc[~adlbc['AVISIT'].str.contains('Baseline', case=False, na=False)].copy()
    
    # Pivot baseline labs
    baseline_pivot = adlbc_baseline.pivot_table(
        index='USUBJID',
        columns='PARAMCD',
        values='AVAL',
        aggfunc='first'
    )
    
    # Pivot post-treatment labs (use first post-baseline visit)
    post_pivot = adlbc_post.sort_values(['USUBJID', 'ADY']).groupby('USUBJID').first().pivot_table(
        index='USUBJID',
        columns='PARAMCD',
        values='AVAL',
        aggfunc='first'
    )
    
    # Merge with subject data
    adsl_key = adsl[['USUBJID', 'SUBJID', 'AGE', 'SEX', 'TRT01P', 'BMIBL', 'HEIGHTBL', 'WEIGHTBL']].copy()
    
    # Merge baseline labs
    adsl_key = adsl_key.merge(baseline_pivot, left_on='USUBJID', right_index=True, how='inner')
    
    # Merge post-treatment labs
    adsl_key = adsl_key.merge(post_pivot, left_on='USUBJID', right_index=True, how='left', suffixes=('_baseline', '_post'))
    
    # Calculate deltas for each lab parameter
    samples = []
    
    for _, row in adsl_key.iterrows():
        # Get drug name and normalize it
        raw_drug_name = str(row['TRT01P']).strip()
        
        # Skip Placebo (no SMILES, not useful for training)
        if raw_drug_name.lower() == 'placebo':
            continue
        
        # Map drug name to one with SMILES
        mapped_drug_name = raw_drug_name
        
        # Try exact match first
        if raw_drug_name in drug_smiles:
            if drug_smiles[raw_drug_name] is None:
                continue  # Skip if SMILES is null
            # Use exact match
            mapped_drug_name = raw_drug_name
        else:
            # Try base name mapping (e.g., "Xanomeline" -> "Xanomeline Low Dose")
            base_name = raw_drug_name.split()[0] if ' ' in raw_drug_name else raw_drug_name
            if base_name in drug_name_mapping:
                mapped_drug_name = drug_name_mapping[base_name]
            else:
                # Skip drugs without SMILES mapping
                continue
        
        sample = {
            'patient_id': row['SUBJID'],
            'drug_name': mapped_drug_name,  # Use mapped name
            'age': float(row['AGE']) if pd.notna(row['AGE']) else 50.0,
            'sex': 'M' if str(row['SEX']).upper() == 'M' else 'F',
            'bmi': float(row['BMIBL']) if pd.notna(row['BMIBL']) else 25.0,
        }
        
        # Calculate deltas for each lab
        for cdisc_code, lab_feat in CDISC_TO_LAB_MAP.items():
            baseline_col = f'{cdisc_code}_baseline'
            post_col = f'{cdisc_code}_post'
            
            baseline_val = row.get(baseline_col)
            post_val = row.get(post_col)
            
            if pd.notna(baseline_val) and pd.notna(post_val):
                delta = float(post_val) - float(baseline_val)
                sample[f'{lab_feat}_delta'] = delta
                sample[f'{lab_feat}_baseline'] = float(baseline_val)
        
        # Only include samples with at least one lab delta
        if any(k.endswith('_delta') for k in sample.keys()):
            samples.append(sample)
    
    df = pd.DataFrame(samples)
    print(f"  ✓ Extracted {len(df):,} real patient-drug-response samples")
    print(f"  ✓ Unique drugs: {df['drug_name'].nunique()}")
    print(f"  ✓ Unique patients: {df['patient_id'].nunique()}")
    
    return df


def augment_real_data(
    real_df: pd.DataFrame,
    n_augmentations_per_sample: int = 10,
    noise_std: float = 0.1,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    seed: int = 42
) -> pd.DataFrame:
    """
    Augment real data by adding variations.
    
    Args:
        real_df: DataFrame with real patient-drug-response data
        n_augmentations_per_sample: Number of augmentations per original sample
        noise_std: Standard deviation of Gaussian noise (as fraction of value)
        scale_range: Range for random scaling of lab deltas
        seed: Random seed
        
    Returns:
        DataFrame with ONLY augmentations (original samples excluded)
    """
    np.random.seed(seed)
    
    print("\n" + "="*80)
    print("AUGMENTING REAL DATA")
    print("="*80)
    print(f"\n  Original samples: {len(real_df):,}")
    print(f"  Augmentations per sample: {n_augmentations_per_sample}")
    print(f"  Noise std: {noise_std}")
    print(f"  Scale range: {scale_range}")
    
    augmented_samples = []
    
    for idx, row in real_df.iterrows():
        for aug_idx in range(n_augmentations_per_sample):
            aug_sample = row.copy()
            
            # Add noise to demographics
            aug_sample['age'] = max(18, min(90, row['age'] + np.random.normal(0, 2.0)))
            aug_sample['bmi'] = max(15, min(50, row['bmi'] * np.random.uniform(0.95, 1.05)))
            
            # Augment lab deltas and baselines
            for col in real_df.columns:
                if col.endswith('_delta'):
                    # Scale delta with random factor
                    scale = np.random.uniform(scale_range[0], scale_range[1])
                    original_delta = row[col]
                    if pd.notna(original_delta):
                        # Add noise proportional to magnitude
                        noise = np.random.normal(0, abs(original_delta) * noise_std)
                        aug_sample[col] = original_delta * scale + noise
                
                elif col.endswith('_baseline'):
                    # Add small noise to baseline
                    original_baseline = row[col]
                    if pd.notna(original_baseline):
                        noise = np.random.normal(0, abs(original_baseline) * noise_std * 0.5)
                        aug_sample[col] = original_baseline + noise
            
            # Mark as augmented (for tracking)
            aug_sample['_is_augmented'] = True
            aug_sample['_original_idx'] = idx
            
            augmented_samples.append(aug_sample)
    
    aug_df = pd.DataFrame(augmented_samples)
    print(f"\n  ✓ Generated {len(aug_df):,} augmented samples")
    print(f"  ✓ Augmentation ratio: {len(aug_df) / len(real_df):.1f}x")
    
    return aug_df


def balance_augmented_data(
    augmented_df: pd.DataFrame,
    target_size: int,
    drug_balance: bool = True,
    seed: int = 42
) -> pd.DataFrame:
    """
    Balance augmented data to match target size.
    
    Args:
        augmented_df: DataFrame with augmented samples
        target_size: Target number of samples
        drug_balance: If True, balance across drugs proportionally
        seed: Random seed
        
    Returns:
        Balanced DataFrame
    """
    np.random.seed(seed)
    
    print("\n" + "="*80)
    print("BALANCING AUGMENTED DATA")
    print("="*80)
    print(f"\n  Original augmented samples: {len(augmented_df):,}")
    print(f"  Target size: {target_size:,}")
    
    if len(augmented_df) <= target_size:
        print(f"  ✓ No balancing needed (already <= target)")
        return augmented_df
    
    if drug_balance:
        # Balance proportionally across drugs
        drug_counts = augmented_df['drug_name'].value_counts()
        total_samples = len(augmented_df)
        
        balanced_samples = []
        for drug, count in drug_counts.items():
            # Proportion of this drug
            proportion = count / total_samples
            target_for_drug = int(target_size * proportion)
            
            # Sample from this drug's augmentations
            drug_df = augmented_df[augmented_df['drug_name'] == drug]
            if len(drug_df) > target_for_drug:
                sampled = drug_df.sample(n=target_for_drug, random_state=seed)
            else:
                sampled = drug_df
            
            balanced_samples.append(sampled)
        
        balanced_df = pd.concat(balanced_samples, ignore_index=True)
        
        # If still too large, randomly sample
        if len(balanced_df) > target_size:
            balanced_df = balanced_df.sample(n=target_size, random_state=seed).reset_index(drop=True)
    else:
        # Simple random sampling
        balanced_df = augmented_df.sample(n=target_size, random_state=seed).reset_index(drop=True)
    
    print(f"  ✓ Balanced to {len(balanced_df):,} samples")
    print(f"  ✓ Drugs represented: {balanced_df['drug_name'].nunique()}")
    
    return balanced_df


def prepare_real_data_for_training(
    adsl_path: str = 'data/cdisc/adsl.csv',
    adlbc_path: str = 'data/cdisc/adlbc.csv',
    drug_smiles_path: str = 'data/mappings/drug_smiles_mapping.json',
    n_augmentations_per_sample: int = 10,
    target_size: Optional[int] = None,
    noise_std: float = 0.1,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    seed: int = 42
) -> pd.DataFrame:
    """
    Complete pipeline: Extract, augment, and balance real CDISC data.
    
    Returns ONLY augmentations (original data excluded).
    
    Args:
        adsl_path: Path to adsl.csv
        adlbc_path: Path to adlbc.csv
        drug_smiles_path: Path to drug SMILES mapping
        n_augmentations_per_sample: Number of augmentations per original
        target_size: Target number of augmented samples (None = use all)
        noise_std: Noise standard deviation
        scale_range: Range for scaling augmentations
        seed: Random seed
        
    Returns:
        DataFrame with augmented real data ready for training
    """
    # Extract real data
    real_df = extract_real_cdisc_data(adsl_path, adlbc_path, drug_smiles_path)
    
    if len(real_df) == 0:
        warnings.warn("No real data extracted. Returning empty DataFrame.")
        return pd.DataFrame()
    
    # Augment (ONLY augmentations, no originals)
    augmented_df = augment_real_data(
        real_df,
        n_augmentations_per_sample=n_augmentations_per_sample,
        noise_std=noise_std,
        scale_range=scale_range,
        seed=seed
    )
    
    # Balance if target size specified
    if target_size is not None:
        augmented_df = balance_augmented_data(
            augmented_df,
            target_size=target_size,
            drug_balance=True,
            seed=seed
        )
    
    # Remove tracking columns
    if '_is_augmented' in augmented_df.columns:
        augmented_df = augmented_df.drop(columns=['_is_augmented', '_original_idx'])
    
    print("\n" + "="*80)
    print("REAL DATA PREPARATION COMPLETE")
    print("="*80)
    print(f"\n  ✓ Final augmented samples: {len(augmented_df):,}")
    print(f"  ✓ Ready for training (ONLY augmentations, no originals)")
    
    return augmented_df

