"""Comprehensive NHANES preprocessing: Remove duplicates, combine columns, create canonical dataset.

This script:
1. Removes duplicate rows
2. Combines duplicate columns (same feature, different labs/cycles) into canonical columns
3. Keeps only drug-testing relevant features
4. Handles missing values appropriately
5. Saves clean dataset ready for VAE training
"""
import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import defaultdict
import json

# Paths
BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / 'raw_data' / 'merged_nhanes.csv'
OUTPUT_DIR = BASE_DIR / 'preprocessed_nhanes'
OUTPUT_CSV = OUTPUT_DIR / 'nhanes_clean.csv'
METADATA_JSON = OUTPUT_DIR / 'preprocessing_metadata.json'

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COMPREHENSIVE NHANES PREPROCESSING")
print("="*80)

# ============================================================================
# STEP 1: LOAD RAW DATA
# ============================================================================
print("\n[1/8] Loading raw data...")
df = pd.read_csv(RAW_CSV)
print(f"  Original shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

original_shape = df.shape
original_cols = list(df.columns)

# ============================================================================
# STEP 2: REMOVE DUPLICATE ROWS
# ============================================================================
print("\n[2/8] Removing duplicate rows...")
n_before = len(df)
df = df.drop_duplicates()
n_after = len(df)
n_duplicates_removed = n_before - n_after
print(f"  Removed {n_duplicates_removed:,} duplicate rows")
print(f"  Remaining: {n_after:,} rows")

# ============================================================================
# STEP 3: IDENTIFY AND COMBINE DUPLICATE COLUMNS
# ============================================================================
print("\n[3/8] Identifying duplicate columns (same feature, different labs/cycles)...")

# Map columns to their base names
# Example: LBXCRP_CRP, LBXCRP_L11, LBXCRP_LAB11 → LBXCRP
column_groups = defaultdict(list)

for col in df.columns:
    # Remove common NHANES lab/cycle suffixes
    base = re.sub(
        r'_(CRP|L\d+|L\d+[A-Z]+|L\d+_[A-Z]|LAB\d+|LAB\d+_[A-Z]|BIOPRO|THYROD|FERTIN|'
        r'GLU|IN|PBCD|VID|BMX|BMX_[A-Z]|DEMO|ALQ|ALQ_[A-Z]|SMQ|SMQRTU|SMQMEC|'
        r'MCQ|MCQ_[A-Z]|DIQ|PFQ|PAQ|PAQ_[A-Z]|DBQ|DBQ_[A-Z]|DUQ|KIQ|KIQ_U|'
        r'HSQ|HSQ_[A-Z]|RHQ|ALB_CR|ALB_CR_F|CBC|L40|L40FE|L06|L06_[A-Z]|L06BMT_C|'
        r'L10|L10AM|L11|L13|L13_[A-Z]|L13AM|L16|L18|L25|RXQ_RX)$', 
        '', col
    )
    column_groups[base].append(col)

# Find groups with multiple columns (duplicates)
duplicate_groups = {k: v for k, v in column_groups.items() if len(v) > 1}
print(f"  Found {len(duplicate_groups)} features with multiple variants")

# Show top duplicates
sorted_dups = sorted(duplicate_groups.items(), key=lambda x: len(x[1]), reverse=True)
print(f"\n  Top 20 duplicated features:")
for i, (base, variants) in enumerate(sorted_dups[:20], 1):
    print(f"    {i}. {base}: {len(variants)} variants → {', '.join(variants[:3])}...")

# ============================================================================
# STEP 4: COMBINE DUPLICATE COLUMNS
# ============================================================================
print("\n[4/8] Combining duplicate columns...")

columns_to_drop = []
columns_created = []

for base_name, variant_cols in duplicate_groups.items():
    # Strategy: Combine using coalesce (first non-null value across variants)
    # This preserves data from all lab cycles
    
    # Create combined column
    combined = df[variant_cols[0]].copy()
    for col in variant_cols[1:]:
        combined = combined.fillna(df[col])
    
    # Add combined column with base name
    df[base_name] = combined
    columns_created.append(base_name)
    
    # Mark variants for deletion
    columns_to_drop.extend(variant_cols)

print(f"  Created {len(columns_created)} canonical columns")
print(f"  Marked {len(columns_to_drop)} duplicate columns for removal")

# Remove duplicate columns
df = df.drop(columns=columns_to_drop)
print(f"  New shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

# ============================================================================
# STEP 5: FILTER TO DRUG-TESTING RELEVANT FEATURES
# ============================================================================
print("\n[5/8] Filtering to drug-testing relevant features...")

# Define patterns for features to KEEP
keep_patterns = [
    # ID
    r'^SEQN$',
    
    # User inputs (demographics - core 4)
    r'^RIDAGEEX$', r'^DMDHRGND$', r'^BMXHT$', r'^BMXWT$',
    
    # Blood labs - glucose/insulin
    r'^LBXGLU', r'^LBXIN', r'^LBXSGL',
    
    # Blood labs - lipids
    r'^LBXTC', r'^LBXTR', r'^LBXLDL', r'^LBXHDL', r'^LBXSCH', r'^LBDHDD', r'^LBDLDL', r'^LBDTC',
    
    # Blood labs - liver function
    r'^LBXSAS', r'^LBXSAL', r'^LBXSAP', r'^LBXSBU', r'^LBXSTP', r'^LBXSALB', r'^LBXSGB', r'^LBXSGT', r'^LBXSBI',
    
    # Blood labs - kidney function
    r'^LBXSCR', r'^LBXSUA', r'^LBDSC', r'^URXUCR', r'^URXUMA', r'^URXACT',
    
    # Blood labs - CBC
    r'^LBXWBC', r'^LBXRBC', r'^LBXHGB', r'^LBXHCT', r'^LBXMCV', r'^LBXMCH', r'^LBXMC',
    r'^LBXPLT', r'^LBXLYPCT', r'^LBXMOPCT', r'^LBXNEPCT', r'^LBXEOPCT', r'^LBXBAPCT',
    r'^LBXLYMNO', r'^LBXMONO', r'^LBXNEO', r'^LBXEONO', r'^LBXBANO',
    
    # Blood labs - iron/vitamins
    r'^LBXIRN', r'^LBXFER', r'^LBXFOL', r'^LBXB12', r'^LBXTIB', r'^LBDPCT',
    r'^LBXVE', r'^LBDVE', r'^LBXVIA', r'^LBXVIC',
    
    # Blood labs - inflammation
    r'^LBXCRP', r'^LBXHCY',
    
    # Blood labs - thyroid
    r'^LBXTT', r'^LBXTSH',
    
    # Blood labs - electrolytes
    r'^LBXSNA', r'^LBXSK', r'^LBXSCL', r'^LBXSCA', r'^LBXSPH', r'^LBXSOSSI',
    
    # Blood labs - heavy metals
    r'^LBXBPB', r'^LBXBCD', r'^LBXTHG', r'^LBXIHG', r'^LBXBSE', r'^LBXBMN',
    
    # Body measurements
    r'^BMXBMI', r'^BMXLEG', r'^BMXARML', r'^BMXARMC', r'^BMXWAIST', r'^BMXHIP',
    r'^BMXTRI', r'^BMXSUB', r'^BMXTHICR', r'^BMXSAD',
    
    # Blood pressure
    r'^BPXSY', r'^BPXDI', r'^BPXPLS',
    
    # Essential conditioning - race/ethnicity
    r'^RIDRETH',
    
    # Essential conditioning - disease history
    r'^DIQ010', r'^MCQ160', r'^KIQ022', r'^MCQ220',
    
    # Essential conditioning - smoking/alcohol
    r'^SMQ020', r'^SMQ040', r'^ALQ120', r'^ALQ130',
]

# Identify columns to keep
keep_cols = set()
for pattern in keep_patterns:
    matched = [col for col in df.columns if re.search(pattern, col, re.IGNORECASE)]
    keep_cols.update(matched)

keep_cols = sorted(list(keep_cols))

print(f"  Identified {len(keep_cols)} relevant columns to keep")
print(f"  Dropping {len(df.columns) - len(keep_cols)} irrelevant columns")

# Filter dataframe
df = df[keep_cols]
print(f"  Filtered shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

# ============================================================================
# STEP 6: HANDLE MISSING VALUES
# ============================================================================
print("\n[6/8] Handling missing values...")

# Calculate missing percentages
missing_pct = (df.isnull().sum() / len(df)) * 100

# Drop columns with >80% missing
high_missing = missing_pct[missing_pct > 80].index.tolist()
print(f"  Dropping {len(high_missing)} columns with >80% missing data")
df = df.drop(columns=high_missing)

# Drop rows with >50% missing values
row_missing_pct = (df.isnull().sum(axis=1) / len(df.columns)) * 100
rows_to_drop = row_missing_pct[row_missing_pct > 50].index
print(f"  Dropping {len(rows_to_drop):,} rows with >50% missing values")
df = df.drop(index=rows_to_drop)

# Impute remaining missing values
print(f"  Imputing remaining missing values...")
for col in df.columns:
    if col == 'SEQN':
        continue
    
    if df[col].dtype in ['float64', 'int64']:
        # Numeric: impute with median
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    else:
        # Categorical: impute with mode
        if df[col].isnull().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
            df[col] = df[col].fillna(mode_val)

print(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
print(f"  Remaining missing values: {df.isnull().sum().sum()}")

# ============================================================================
# STEP 7: CONVERT AGE FROM MONTHS TO YEARS (if needed)
# ============================================================================
print("\n[7/8] Processing age column...")
if 'RIDAGEEX' in df.columns:
    # Check if age is in months (values > 120 suggest months)
    if df['RIDAGEEX'].max() > 120:
        print(f"  Converting RIDAGEEX from months to years")
        print(f"    Range before: {df['RIDAGEEX'].min():.1f} - {df['RIDAGEEX'].max():.1f} months")
        df['RIDAGEEX'] = df['RIDAGEEX'] / 12
        print(f"    Range after: {df['RIDAGEEX'].min():.1f} - {df['RIDAGEEX'].max():.1f} years")
    else:
        print(f"  RIDAGEEX already in years (range: {df['RIDAGEEX'].min():.1f} - {df['RIDAGEEX'].max():.1f})")

# ============================================================================
# STEP 8: SAVE CLEANED DATASET
# ============================================================================
print("\n[8/8] Saving cleaned dataset...")

df.to_csv(OUTPUT_CSV, index=False)
print(f"  ✓ Saved: {OUTPUT_CSV}")
print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
print(f"    Size: {OUTPUT_CSV.stat().st_size / 1e6:.1f} MB")

# Save metadata
metadata = {
    'preprocessing_date': pd.Timestamp.now().isoformat(),
    'source_file': str(RAW_CSV),
    'output_file': str(OUTPUT_CSV),
    'steps_performed': [
        '1. Removed duplicate rows',
        '2. Combined duplicate columns (same feature, different labs/cycles)',
        '3. Filtered to drug-testing relevant features',
        '4. Dropped columns with >80% missing',
        '5. Dropped rows with >50% missing',
        '6. Imputed remaining missing values (median for numeric, mode for categorical)',
        '7. Converted age from months to years',
    ],
    'original_shape': {'rows': original_shape[0], 'columns': original_shape[1]},
    'final_shape': {'rows': df.shape[0], 'columns': df.shape[1]},
    'duplicate_rows_removed': n_duplicates_removed,
    'duplicate_column_groups': len(duplicate_groups),
    'canonical_columns_created': len(columns_created),
    'high_missing_columns_dropped': len(high_missing),
    'sparse_rows_dropped': len(rows_to_drop),
    'feature_categories': {
        'user_inputs': 4,  # age, sex, height, weight
        'blood_labs': len([c for c in df.columns if c.startswith('LBX') or c.startswith('LBD') or c.startswith('URX')]),
        'body_measurements': len([c for c in df.columns if c.startswith('BMX') or c.startswith('BPX')]),
        'conditioning': len([c for c in df.columns if any(c.startswith(p) for p in ['RIDRETH', 'DIQ', 'MCQ', 'KIQ', 'SMQ', 'ALQ'])]),
    },
    'final_columns': list(df.columns),
}

with open(METADATA_JSON, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  ✓ Saved metadata: {METADATA_JSON}")

# Print summary
print("\n" + "="*80)
print("PREPROCESSING COMPLETE")
print("="*80)
print(f"\nSummary:")
print(f"  Original: {original_shape[0]:,} rows × {original_shape[1]:,} columns")
print(f"  Final: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
print(f"  Rows removed: {original_shape[0] - df.shape[0]:,} ({(original_shape[0] - df.shape[0])/original_shape[0]*100:.1f}%)")
print(f"  Columns removed: {original_shape[1] - df.shape[1]:,} ({(original_shape[1] - df.shape[1])/original_shape[1]*100:.1f}%)")
print(f"\nFeature breakdown:")
print(f"  User inputs: {metadata['feature_categories']['user_inputs']}")
print(f"  Blood labs: {metadata['feature_categories']['blood_labs']}")
print(f"  Body measurements: {metadata['feature_categories']['body_measurements']}")
print(f"  Conditioning: {metadata['feature_categories']['conditioning']}")
print(f"\nOutput files:")
print(f"  - {OUTPUT_CSV}")
print(f"  - {METADATA_JSON}")
print(f"\nNext step: Update constants.py to use the new dataset")
print(f"  PREPROC_CSV = PREPROC_DIR / 'nhanes_clean.csv'")
