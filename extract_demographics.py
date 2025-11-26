"""
Extract age and sex from nhanes_preprocessed.csv and merge into nhanes_final.csv
"""
import pandas as pd
import numpy as np

print("=" * 80)
print("EXTRACTING DEMOGRAPHICS FROM PREPROCESSED NHANES")
print("=" * 80)

# Load raw merged data with demographics (has all age cycles)
print("\n1. Loading raw merged data with demographics...")

# Try to find age in years (RIDAGEYR) from any cycle
age_cols = [c for c in pd.read_csv('raw_data/merged_nhanes.csv', nrows=1).columns 
            if 'RIDAGEYR' in c and 'DEMO' in c]
sex_cols = [c for c in pd.read_csv('raw_data/merged_nhanes.csv', nrows=1).columns 
            if 'RIAGENDR' in c and 'DEMO' in c]

print(f"   Found {len(age_cols)} age columns, {len(sex_cols)} sex columns")
print(f"   Using: {age_cols[0]}, {sex_cols[0]}")

# Load with coalesce across all cycles
preproc = pd.read_csv('raw_data/merged_nhanes.csv', usecols=['SEQN'] + age_cols + sex_cols)
print(f"   Loaded {len(preproc):,} rows")

# Coalesce age across all cycles (take first non-null)
print("\n2. Coalescing age across NHANES cycles...")
preproc['AGE'] = preproc[age_cols].bfill(axis=1).iloc[:, 0]
print(f"   Non-null: {preproc['AGE'].notna().sum():,}")
print(f"   Range: {preproc['AGE'].min():.1f} - {preproc['AGE'].max():.1f} years")
print(f"   Mean: {preproc['AGE'].mean():.1f} years")

# Map sex (1=Male, 2=Female) to match expected format
print("\n3. Coalescing sex across NHANES cycles...")
preproc['SEX_CODE'] = preproc[sex_cols].bfill(axis=1).iloc[:, 0]
preproc['SEX'] = preproc['SEX_CODE'].map({1.0: 'M', 2.0: 'F'})
print(f"   Non-null: {preproc['SEX'].notna().sum():,}")
print(f"   Male: {(preproc['SEX'] == 'M').sum():,}")
print(f"   Female: {(preproc['SEX'] == 'F').sum():,}")

# Load the cleaned dataset
print("\n4. Loading cleaned dataset...")
cleaned = pd.read_csv('preprocessed_nhanes/nhanes_final.csv')
print(f"   Shape: {cleaned.shape[0]:,} rows × {cleaned.shape[1]} columns")
print(f"   Unique SEQN: {cleaned['SEQN'].nunique():,}")

# Merge demographics
print("\n5. Merging demographics...")

# Deduplicate raw data first (multiple cycles create duplicates)
print("   Deduplicating raw data by SEQN (keeping first occurrence)...")
preproc_dedup = preproc.drop_duplicates(subset='SEQN', keep='first')
print(f"   Deduplicated: {len(preproc):,} → {len(preproc_dedup):,} rows")

result = cleaned.merge(
    preproc_dedup[['SEQN', 'AGE', 'SEX']], 
    on='SEQN', 
    how='left'
)
print(f"   Merged shape: {result.shape[0]:,} rows × {result.shape[1]} columns")

# Check for missing values
print("\n6. Checking for missing values...")
missing_age = result['AGE'].isna().sum()
missing_sex = result['SEX'].isna().sum()
print(f"   Missing AGE: {missing_age:,} ({100*missing_age/len(result):.1f}%)")
print(f"   Missing SEX: {missing_sex:,} ({100*missing_sex/len(result):.1f}%)")

# Reorder columns: SEQN, AGE, SEX, then user inputs, then rest
user_input_cols = ['AGE', 'SEX', 'BMXHT', 'BMXWT']
other_cols = [c for c in result.columns if c not in ['SEQN'] + user_input_cols]
final_cols = ['SEQN'] + user_input_cols + other_cols

result = result[final_cols]

# Save
print("\n7. Saving complete dataset...")
output_path = 'preprocessed_nhanes/nhanes_final_complete.csv'
result.to_csv(output_path, index=False)
print(f"   ✓ Saved to: {output_path}")

# Summary statistics for user inputs
print("\n" + "=" * 80)
print("FINAL DATASET SUMMARY - USER INPUTS")
print("=" * 80)
for col in user_input_cols:
    if result[col].dtype == 'object':
        print(f"\n{col}:")
        print(f"  Values: {result[col].value_counts().to_dict()}")
    else:
        print(f"\n{col}:")
        print(f"  Count: {result[col].notna().sum():,}")
        print(f"  Range: {result[col].min():.1f} - {result[col].max():.1f}")
        print(f"  Mean: {result[col].mean():.1f}")
        print(f"  Median: {result[col].median():.1f}")

print("\n" + "=" * 80)
print(f"✓ COMPLETE! Dataset ready for CVAE training")
print(f"  Shape: {result.shape[0]:,} rows × {result.shape[1]} columns")
print(f"  4 User Inputs: AGE ✓, SEX ✓, BMXHT ✓, BMXWT ✓")
print("=" * 80)
