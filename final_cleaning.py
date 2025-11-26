"""Final cleaning: Merge duplicate lab features, remove unnecessary columns, create canonical dataset.

This script:
1. Merges duplicate lab features (SI units vs conventional units → keep one canonical)
2. Removes unnecessary survey/questionnaire columns
3. Ensures no duplicate rows
4. Keeps only drug-testing relevant features
5. Adds missing essential features (age, sex, race if available)
6. Saves final clean dataset ready for CVAE training
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Paths
BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / 'preprocessed_nhanes' / 'nhanes_clean.csv'
OUTPUT_CSV = BASE_DIR / 'preprocessed_nhanes' / 'nhanes_cleaned.csv'
METADATA_JSON = BASE_DIR / 'preprocessed_nhanes' / 'final_cleaning_metadata.json'

print("="*80)
print("FINAL NHANES DATASET CLEANING")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/7] Loading partially cleaned data...")
df = pd.read_csv(INPUT_CSV)
print(f"  Input shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
print(f"\n  Current columns:")
for i, col in enumerate(df.columns, 1):
    print(f"    {i}. {col}")

original_shape = df.shape
original_cols = list(df.columns)

# ============================================================================
# STEP 2: REMOVE DUPLICATE ROWS
# ============================================================================
print("\n[2/7] Removing duplicate rows...")
n_before = len(df)
df = df.drop_duplicates()
n_after = len(df)
n_duplicates = n_before - n_after
print(f"  Removed {n_duplicates:,} duplicate rows")
print(f"  Remaining: {n_after:,} rows")

# ============================================================================
# STEP 3: MERGE DUPLICATE LAB FEATURES (SI vs Conventional Units)
# ============================================================================
print("\n[3/7] Merging duplicate lab features...")

# Define pairs: (SI_column, conventional_column, canonical_name)
# Strategy: Merge SI and conventional units into one column, prefer non-null values
duplicate_pairs = [
    # Urine creatinine: URXUCR (mg/dL) vs URXUCRSI (μmol/L) → keep URXUCR
    ('URXUCRSI', 'URXUCR', 'URXUCR'),
    
    # Urine microalbumin: URXUMA (μg/mL) vs URXUMASI (mg/L) → keep URXUMA
    ('URXUMASI', 'URXUMA', 'URXUMA'),
    
    # Creatinine: LBDSCRSI (SI) vs conventional → keep LBDSCR
    ('LBDSCRSI', None, 'LBDSCR'),  # Create LBDSCR if doesn't exist
    
    # Cholesterol: LBDSCHSI (SI) vs conventional → keep LBDSCH
    ('LBDSCHSI', None, 'LBDSCH'),
    
    # AST: LBXSASSI (SI) vs conventional → keep LBXSAS
    ('LBXSASSI', None, 'LBXSAS'),
    
    # GGT: LBXSGTSI (SI) vs conventional → keep LBXSGT
    ('LBXSGTSI', None, 'LBXSGT'),
    
    # Potassium: LBXSKSI (SI) vs conventional → keep LBXSK
    ('LBXSKSI', None, 'LBXSK'),
    
    # Sodium: LBXSNASI (SI) vs conventional → keep LBXSNA
    ('LBXSNASI', None, 'LBXSNA'),
    
    # Chloride: LBXSCLSI (SI) vs conventional → keep LBXSCL
    ('LBXSCLSI', None, 'LBXSCL'),
    
    # Osmolality: LBXSOSSI (SI) vs conventional → keep LBXSOS
    ('LBXSOSSI', None, 'LBXSOS'),
    
    # Total cholesterol: LBDTCSI (SI) vs conventional → keep LBDTC
    ('LBDTCSI', None, 'LBDTC'),
]

merged_features = []
columns_to_drop = []

for si_col, conv_col, canonical_name in duplicate_pairs:
    if si_col in df.columns:
        # If conventional column exists, merge
        if conv_col and conv_col in df.columns:
            # Coalesce: take first non-null value
            df[canonical_name] = df[conv_col].fillna(df[si_col])
            columns_to_drop.extend([si_col, conv_col])
            merged_features.append(f"{si_col} + {conv_col} → {canonical_name}")
        else:
            # Just rename SI to canonical
            df[canonical_name] = df[si_col]
            columns_to_drop.append(si_col)
            merged_features.append(f"{si_col} → {canonical_name}")
    elif conv_col and conv_col in df.columns:
        # Only conventional exists, rename to canonical
        df[canonical_name] = df[conv_col]
        columns_to_drop.append(conv_col)
        merged_features.append(f"{conv_col} → {canonical_name}")

# Remove duplicate columns
columns_to_drop = list(set(columns_to_drop))
df = df.drop(columns=columns_to_drop, errors='ignore')

print(f"  Merged {len(merged_features)} duplicate lab features:")
for merge in merged_features:
    print(f"    - {merge}")

print(f"  Dropped {len(columns_to_drop)} duplicate columns")
print(f"  New shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

# ============================================================================
# STEP 4: REMOVE UNNECESSARY SURVEY/QUESTIONNAIRE COLUMNS
# ============================================================================
print("\n[4/7] Removing unnecessary survey/questionnaire columns...")

# Define columns to remove (not relevant for drug testing)
columns_to_remove = []

# Alcohol questionnaire - too detailed, not relevant for acute drug effects
alq_cols = [col for col in df.columns if col.startswith('ALQ')]
columns_to_remove.extend(alq_cols)
print(f"  Removing {len(alq_cols)} alcohol questionnaire columns: {alq_cols}")

# Medical conditions questionnaire - keep only essential disease history
# Remove: MCQ160A (arthritis), MCQ160D (emphysema), MCQ160G (weak kidneys), 
#         MCQ160M (thyroid), MCQ160N (other liver)
# Keep: MCQ160B (CHF), MCQ160C (CHD), MCQ160E (heart attack), MCQ160F (stroke),
#       MCQ160K (COPD), MCQ160L (liver condition), MCQ220 (cancer)
mcq_to_remove = ['MCQ160A', 'MCQ160D', 'MCQ160G', 'MCQ160M', 'MCQ160N']
mcq_to_remove = [col for col in mcq_to_remove if col in df.columns]
columns_to_remove.extend(mcq_to_remove)
print(f"  Removing {len(mcq_to_remove)} non-essential MCQ columns: {mcq_to_remove}")

# Drop columns
df = df.drop(columns=columns_to_remove, errors='ignore')
print(f"  Total columns removed: {len(columns_to_remove)}")
print(f"  New shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

# ============================================================================
# STEP 5: ADD MISSING ESSENTIAL FEATURES
# ============================================================================
print("\n[5/7] Checking for essential user-input features...")

# Check for age, sex, height, weight
essential_features = {
    'RIDAGEEX': 'Age (years)',
    'DMDHRGND': 'Sex (1=M, 2=F)',
    'BMXHT': 'Height (cm)',
    'BMXWT': 'Weight (kg)',
}

missing_essential = []
for feat, desc in essential_features.items():
    if feat not in df.columns:
        missing_essential.append((feat, desc))

if missing_essential:
    print(f"  ⚠ WARNING: Missing {len(missing_essential)} essential features:")
    for feat, desc in missing_essential:
        print(f"    - {feat}: {desc}")
    print(f"\n  These features must be added from the original raw data!")
    print(f"  Current dataset only has: {[c for c in df.columns if c.startswith('BMX')]}")
else:
    print(f"  ✓ All 4 essential user-input features present")

# Check for race/ethnicity
if 'RIDRETH1' not in df.columns and 'RIDRETH3' not in df.columns:
    print(f"  ⚠ WARNING: Missing race/ethnicity (RIDRETH1 or RIDRETH3)")
else:
    print(f"  ✓ Race/ethnicity feature present")

# ============================================================================
# STEP 6: RENAME COLUMNS TO CANONICAL NAMES
# ============================================================================
print("\n[6/7] Standardizing column names...")

# Map current names to more descriptive canonical names (optional)
# For now, keep NHANES codes as-is for consistency
column_mapping = {
    # Already canonical
}

if column_mapping:
    df = df.rename(columns=column_mapping)
    print(f"  Renamed {len(column_mapping)} columns")
else:
    print(f"  Keeping original NHANES column names")

# ============================================================================
# STEP 7: SAVE FINAL CLEANED DATASET
# ============================================================================
print("\n[7/7] Saving final cleaned dataset...")

# Sort columns: SEQN first, then alphabetically
cols = df.columns.tolist()
if 'SEQN' in cols:
    cols.remove('SEQN')
    cols = ['SEQN'] + sorted(cols)
    df = df[cols]

df.to_csv(OUTPUT_CSV, index=False)
print(f"  ✓ Saved: {OUTPUT_CSV}")
print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
print(f"    Size: {OUTPUT_CSV.stat().st_size / 1e6:.1f} MB")

# Save metadata
final_cols = list(df.columns)

# Categorize columns
user_input_cols = [c for c in final_cols if c in ['RIDAGEEX', 'DMDHRGND', 'BMXHT', 'BMXWT']]
body_measurement_cols = [c for c in final_cols if c.startswith('BMX') and c not in user_input_cols]
blood_lab_cols = [c for c in final_cols if c.startswith('LBX') or c.startswith('LBD')]
urine_lab_cols = [c for c in final_cols if c.startswith('URX')]
disease_history_cols = [c for c in final_cols if c.startswith('MCQ')]
other_cols = [c for c in final_cols if c not in user_input_cols + body_measurement_cols + 
              blood_lab_cols + urine_lab_cols + disease_history_cols and c != 'SEQN']

metadata = {
    'processing_date': pd.Timestamp.now().isoformat(),
    'source_file': str(INPUT_CSV),
    'output_file': str(OUTPUT_CSV),
    'steps_performed': [
        '1. Removed duplicate rows',
        '2. Merged duplicate lab features (SI vs conventional units)',
        '3. Removed unnecessary survey/questionnaire columns',
        '4. Verified essential user-input features',
        '5. Standardized column names',
        '6. Sorted columns (SEQN first, then alphabetical)',
    ],
    'original_shape': {'rows': original_shape[0], 'columns': original_shape[1]},
    'final_shape': {'rows': df.shape[0], 'columns': df.shape[1]},
    'duplicate_rows_removed': n_duplicates,
    'duplicate_features_merged': len(merged_features),
    'survey_columns_removed': len(columns_to_remove),
    'feature_categories': {
        'user_inputs': len(user_input_cols),
        'body_measurements': len(body_measurement_cols),
        'blood_labs': len(blood_lab_cols),
        'urine_labs': len(urine_lab_cols),
        'disease_history': len(disease_history_cols),
        'other': len(other_cols),
    },
    'user_input_features': user_input_cols,
    'missing_essential_features': [f for f, _ in missing_essential],
    'merged_features': merged_features,
    'removed_columns': columns_to_remove,
    'final_columns': final_cols,
}

with open(METADATA_JSON, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  ✓ Saved metadata: {METADATA_JSON}")

# ============================================================================
# PRINT SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL CLEANING COMPLETE")
print("="*80)

print(f"\nTransformation:")
print(f"  Original: {original_shape[0]:,} rows × {original_shape[1]:,} columns")
print(f"  Final: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
print(f"  Rows removed: {original_shape[0] - df.shape[0]:,}")
print(f"  Columns removed: {original_shape[1] - df.shape[1]:,}")

print(f"\nFeature breakdown:")
print(f"  User inputs: {len(user_input_cols)} - {user_input_cols}")
print(f"  Body measurements: {len(body_measurement_cols)}")
print(f"  Blood labs: {len(blood_lab_cols)}")
print(f"  Urine labs: {len(urine_lab_cols)}")
print(f"  Disease history: {len(disease_history_cols)}")
print(f"  Other: {len(other_cols)}")

print(f"\nFinal columns ({len(final_cols)}):")
for i, col in enumerate(final_cols, 1):
    print(f"  {i:2d}. {col}")

if missing_essential:
    print(f"\n⚠ WARNING: Missing essential features that need to be added:")
    for feat, desc in missing_essential:
        print(f"  - {feat}: {desc}")
    print(f"\nTo fix: Merge with original raw data to get age and sex columns")
else:
    print(f"\n✓ Dataset is ready for CVAE training!")
    print(f"\nNext step: Update constants.py:")
    print(f"  PREPROC_CSV = PREPROC_DIR / 'nhanes_cleaned.csv'")
