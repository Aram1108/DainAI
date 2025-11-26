import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PHASE 1: LOAD AND INITIAL FILTERING
# ============================================================================

class NHANESPreprocessor:
    """
    Comprehensive preprocessing pipeline for NHANES data
    targeting blood-drug interaction simulation modeling.
    """
    
    def __init__(self, coverage_threshold=0.05, variance_threshold=0.01, use_polars=True):
        """
        Parameters:
        -----------
        coverage_threshold : float
            Minimum proportion of non-null values to retain feature (default 5%)
        variance_threshold : float
            Minimum variance to retain feature (default 0.01)
        """
        self.coverage_threshold = coverage_threshold
        self.variance_threshold = variance_threshold
        # Whether to prefer Polars for initial CSV loading (faster for large files)
        self.use_polars = use_polars
        # Maximum allowed proportion missing per row. Rows with a higher
        # fraction of missing values will be dropped early to avoid
        # keeping very sparse samples that add noise.
        self.row_missing_threshold = 0.5
        self.feature_metadata = {}
        self.scaler = StandardScaler()
        self.imputer = IterativeImputer(random_state=42, max_iter=10)
        
    def load_data(self, filepath):
        """
        Load NHANES dataset and perform initial inspection.
        
        Parameters:
        -----------
        filepath : str
            Path to NHANES CSV file
            
        Returns:
        --------
        pd.DataFrame : Loaded dataset
        """
        # If Polars is available and requested, use it for faster CSV reading.
        if self.use_polars:
            try:
                import polars as pl
                print("Using Polars to read CSV (will convert to pandas for processing)...")
                df_pl = pl.read_csv(filepath)
                # Basic stats from Polars without converting whole dataset twice
                n_rows = df_pl.height
                n_cols = len(df_pl.columns)
                print(f"Loaded dataset (polars): {n_rows} observations, {n_cols} features")
                # Convert to pandas for compatibility with the rest of the pipeline
                df = df_pl.to_pandas()
                print(f"Converted polars DataFrame to pandas (shape: {df.shape})")
                return df
            except Exception as e:
                print(f"Polars read failed or not available: {e}. Falling back to pandas.")

        df = pd.read_csv(filepath, low_memory=False)
        print(f"Loaded dataset: {df.shape[0]} observations, {df.shape[1]} features")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        return df
    
    def analyze_coverage(self, df):
        """
        Calculate coverage statistics for each feature.
        
        Returns:
        --------
        pd.DataFrame : Coverage statistics sorted by non-null count
        """
        coverage = pd.DataFrame({
            'feature': df.columns,
            'non_null': df.count(),
            'null': df.isnull().sum(),
            'coverage_pct': (df.count() / len(df)) * 100,
            'dtype': df.dtypes
        }).sort_values('non_null', ascending=False)
        
        return coverage
    
    def filter_by_coverage(self, df, coverage_df):
        """
        Remove features below coverage threshold.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        coverage_df : pd.DataFrame
            Coverage statistics from analyze_coverage()
            
        Returns:
        --------
        pd.DataFrame : Filtered dataset
        list : Names of dropped features
        """
        n_obs = len(df)
        min_non_null = n_obs * self.coverage_threshold
        
        keep_features = coverage_df[
            coverage_df['non_null'] >= min_non_null
        ]['feature'].tolist()

        # Only keep features that still exist in the dataframe. It's possible
        # coverage_df was computed before some aggressive drops, so guard here
        # to avoid KeyError when selecting columns.
        existing_keep = [f for f in keep_features if f in df.columns]
        missing_expected = [f for f in keep_features if f not in df.columns]

        dropped_features = [f for f in df.columns if f not in existing_keep]

        print(f"\n=== Coverage Filtering (threshold={self.coverage_threshold:.1%}) ===")
        print(f"Retained: {len(existing_keep)} features (requested {len(keep_features)})")
        print(f"Dropped: {len(dropped_features)} features")
        if missing_expected:
            print(f"Note: {len(missing_expected)} requested features were not present and were skipped")

        # Safe selection using only existing columns
        return df.loc[:, existing_keep], dropped_features
    
    def drop_irrelevant_categories(self, df):
        """
        Remove feature categories not relevant to blood simulation.
        
        Returns:
        --------
        pd.DataFrame : Filtered dataset
        list : Names of dropped features
        """
        patterns_to_drop = [
            # Survey weights
            r'^WT[A-Z]+\d+YR',
            r'^SDMV',
            
            # Examination comments and quality flags
            r'_CM$', r'EXSTS$', r'EXCMT$',
            
            # Detailed dietary items (keep summary scores only)
            r'^DBQ070[A-D]$', r'^DBQ220[A-D]$', r'^DBD071[A-DU]$', r'^DBD221[A-DU]$',
            r'^DBQ071[A-DU]$',
            
            # Granular physical activity types (keep duration/frequency summaries)
            r'^PAQ724[A-Z]{1,2}$', r'^PAQ759[A-Z]$',
            
            # Amputation and limb-specific measurements
            r'^BMA(AMP|UREXT|UPREL|ULEXT|UPLEL|LOREX|LORKN|LLEXT|LLKNE)',
            r'^BMD(RECUF|SUBF|THICF|LEGF|ARMLF|CALFF)',
            
            # Replicate measurements (keep primary only)
            r'BMXSAD[34]$',
            
            # Laboratory comment fields
            r'LC$',
        ]
        
        dropped = []
        for pattern in patterns_to_drop:
            matches = df.filter(regex=pattern).columns.tolist()
            dropped.extend(matches)
        
        dropped = list(set(dropped))  # Remove duplicates
        df_filtered = df.drop(columns=dropped, errors='ignore')
        
        print(f"\n=== Irrelevant Category Filtering ===")
        print(f"Dropped {len(dropped)} features across {len(patterns_to_drop)} patterns")
        
        return df_filtered, dropped
    
    # ========================================================================
    # PHASE 2: HANDLE REDUNDANCY
    # ========================================================================
    
    def consolidate_unit_duplicates(self, df):
        """
        Keep SI units, drop conventional units for lab values.
        
        Returns:
        --------
        pd.DataFrame : Deduplicated dataset
        dict : Mapping of dropped features to retained equivalents
        """
        # Identify SI vs conventional unit pairs
        # Convention: SI units have "SI" suffix, conventional units don't
        
        si_features = [f for f in df.columns if 'SI' in f.upper()]
        
        unit_map = {}
        dropped = []
        
        for si_feat in si_features:
            # Try to find conventional unit equivalent
            # Example: LBXGLUSI -> LBXGLU
            base = si_feat.replace('SI', '').replace('si', '')
            
            if base in df.columns and base != si_feat:
                unit_map[base] = si_feat
                dropped.append(base)
        
        df_dedup = df.drop(columns=dropped, errors='ignore')
        
        print(f"\n=== Unit Conversion Deduplication ===")
        print(f"Retained {len(si_features)} SI unit features")
        print(f"Dropped {len(dropped)} conventional unit duplicates")
        
        return df_dedup, unit_map
    
    def consolidate_longitudinal_labs(self, df):
        """
        Merge duplicate lab measurements across NHANES cycles.
        
        Strategy: For each analyte, take non-null value with preference for
        most recent cycle (higher suffix letter/number).
        
        Returns:
        --------
        pd.DataFrame : Consolidated dataset
        dict : Mapping of consolidated features
        """
        # Identify lab features with cycle suffixes
        # Pattern: LBXYYY_ZZZZZ where ZZZZZ is cycle identifier
        
        lab_groups = {}
        
        for col in df.columns:
            if col.startswith('LBX') or col.startswith('LBD') or col.startswith('URX'):
                # Extract base analyte name
                parts = col.split('_')
                if len(parts) > 1:
                    base = parts[0]
                    if base not in lab_groups:
                        lab_groups[base] = []
                    lab_groups[base].append(col)
        
        # For features with multiple cycles, consolidate
        consolidated_map = {}
        new_features = {}
        
        for base, variants in lab_groups.items():
            if len(variants) > 1:
                # Sort by cycle (later cycles preferred)
                variants_sorted = sorted(variants)
                
                # Create consolidated feature by coalescing across variants
                # (take first non-null value in sorted order)
                consolidated_name = base + '_CONSOL'
                new_features[consolidated_name] = df[variants_sorted].bfill(axis=1).iloc[:, 0]
                
                consolidated_map[consolidated_name] = variants
        
        # Add new consolidated features
        df_consol = df.copy()
        for name, series in new_features.items():
            df_consol[name] = series
        
        # Drop original variant columns
        to_drop = [v for variants in consolidated_map.values() for v in variants]
        df_consol = df_consol.drop(columns=to_drop, errors='ignore')
        
        print(f"\n=== Longitudinal Lab Consolidation ===")
        print(f"Consolidated {len(consolidated_map)} analytes")
        print(f"Removed {len(to_drop)} duplicate measurements")
        
        return df_consol, consolidated_map

    def consolidate_by_prefix(self, df, prefer_order=None):
        """
        Consolidate columns that represent the same NHANES variable across
        different cycles by using the prefix before the first underscore.

        Strategy: group columns by `prefix = col.split('_')[0]`. If multiple
        variants exist for the same prefix, create a merged column named
        `prefix` (or `prefix_MERGED` if `prefix` already exists) by coalescing
        non-null values (left-to-right preference based on sorted order).

        Returns:
        --------
        pd.DataFrame : DataFrame with merged columns
        dict : Mapping of merged_name -> list of source columns
        """
        from collections import defaultdict

        prefix_groups = defaultdict(list)
        for col in df.columns:
            prefix = col.split('_')[0]
            prefix_groups[prefix].append(col)

        merged_map = {}
        df_new = df.copy()

        for prefix, variants in prefix_groups.items():
            if len(variants) > 1:
                # Determine merged column name: prefer bare prefix if not present
                merged_name = prefix
                if merged_name in df.columns and merged_name not in variants:
                    merged_name = prefix + '_MERGED'

                # Sort deterministically (can be improved to prefer recent cycles)
                variants_sorted = sorted(variants)

                # Coalesce across variants: take first non-null value across columns
                merged_series = df[variants_sorted].bfill(axis=1).iloc[:, 0]

                # Only add merged column if it will actually replace multiple variants
                df_new[merged_name] = merged_series
                merged_map[merged_name] = variants_sorted

        # Drop original variant columns that were merged
        to_drop = [v for variants in merged_map.values() for v in variants if v in df_new.columns and v != v]
        # Note: above list comprehension intentionally results in empty list (avoid auto-drop here)
        # Instead, preserve original columns to be conservative. If user wants to drop originals,
        # they can pass an argument or call a cleanup function later.

        print(f"\n=== Prefix-based Consolidation ===")
        print(f"Created {len(merged_map)} merged features (by prefix)")

        return df_new, merged_map

    def drop_sparse_rows(self, df, threshold=None):
        """
        Drop rows (observations) that have a high proportion of missing values.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        threshold : float or None
            Proportion of allowed missing values per row (0-1). If None,
            uses `self.row_missing_threshold`.

        Returns:
        --------
        pd.DataFrame : DataFrame with sparse rows removed
        list : Dropped row indices
        """
        if threshold is None:
            threshold = self.row_missing_threshold

        row_missing_pct = df.isnull().mean(axis=1)
        to_drop = row_missing_pct[row_missing_pct > threshold].index.tolist()

        print(f"\n=== Row Sparsity Filtering (threshold={threshold:.2f}) ===")
        print(f"Rows to drop: {len(to_drop)} out of {len(df)} ({len(to_drop)/len(df)*100:.2f}%)")

        df_dropped = df.drop(index=to_drop)
        return df_dropped, to_drop
    
    def remove_high_correlation(self, df, threshold=0.95):
        """
        Identify and remove highly correlated feature pairs.
        
        Parameters:
        -----------
        threshold : float
            Correlation coefficient threshold (default 0.95)
            
        Returns:
        --------
        pd.DataFrame : Dataset with redundant features removed
        list : Dropped feature names
        """
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Identify pairs above threshold
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation above threshold
        to_drop = [
            column for column in upper_tri.columns 
            if any(upper_tri[column] > threshold)
        ]
        
        df_reduced = df.drop(columns=to_drop, errors='ignore')
        
        print(f"\n=== High Correlation Filtering (r > {threshold}) ===")
        print(f"Dropped {len(to_drop)} highly correlated features")
        
        return df_reduced, to_drop
    
    # ========================================================================
    # PHASE 3: FEATURE ENGINEERING
    # ========================================================================
    
    def create_composite_features(self, df):
        """
        Engineer composite features from questionnaire data.
        
        Returns:
        --------
        pd.DataFrame : Dataset with new composite features
        dict : Documentation of created features
        """
        df_eng = df.copy()
        composite_docs = {}
        
        # Cardiovascular disease composite
        cvd_components = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']
        available_cvd = [c for c in cvd_components if c in df.columns]
        if available_cvd:
            df_eng['CVD_COMPOSITE'] = df[available_cvd].max(axis=1)
            composite_docs['CVD_COMPOSITE'] = {
                'components': available_cvd,
                'description': 'Any cardiovascular disease (CHF, CHD, angina, MI, stroke)'
            }
        
        # Diabetes severity composite
        if 'DIQ010' in df.columns:
            diabetes_base = df['DIQ010'].fillna(0)
            df_eng['DIABETES_COMPOSITE'] = diabetes_base
            
            # Enhance with insulin use if available
            if 'DIQ050' in df.columns:
                insulin_use = df['DIQ050'].fillna(0)
                df_eng['DIABETES_COMPOSITE'] = df_eng['DIABETES_COMPOSITE'] + insulin_use
                
            composite_docs['DIABETES_COMPOSITE'] = {
                'description': 'Diabetes severity (0=none, 1=diagnosed, 2=on insulin)'
            }
        
        # Liver disease composite
        liver_components = ['MCQ160L']
        available_liver = [c for c in liver_components if c in df.columns]
        if available_liver:
            df_eng['LIVER_DISEASE'] = df[available_liver].max(axis=1)
            composite_docs['LIVER_DISEASE'] = {
                'components': available_liver,
                'description': 'Any liver condition'
            }
        
        # Kidney disease composite
        kidney_components = ['KIQ020', 'KIQ120']
        available_kidney = [c for c in kidney_components if c in df.columns]
        if available_kidney:
            df_eng['KIDNEY_DISEASE'] = df[available_kidney].max(axis=1)
            composite_docs['KIDNEY_DISEASE'] = {
                'components': available_kidney,
                'description': 'Any kidney condition or dialysis'
            }
        
        # Smoking status (3-level: never, former, current)
        smoking_cols = ['SMQ020', 'SMQ040']
        if all(c in df.columns for c in smoking_cols):
            # SMQ020: Ever smoked (1=yes, 2=no)
            # SMQ040: Current smoker (1=every day, 2=some days, 3=not at all)
            df_eng['SMOKING_STATUS'] = 0  # Never
            df_eng.loc[df['SMQ020'] == 1, 'SMOKING_STATUS'] = 1  # Ever
            df_eng.loc[
                (df['SMQ020'] == 1) & (df['SMQ040'].isin([1, 2])), 
                'SMOKING_STATUS'
            ] = 2  # Current
            
            composite_docs['SMOKING_STATUS'] = {
                'description': 'Smoking status (0=never, 1=former, 2=current)'
            }
        
        # Alcohol use category
        alcohol_cols = ['ALQ101', 'ALQ110']
        if 'ALQ101' in df.columns:
            # ALQ101: Had at least 12 drinks in lifetime
            df_eng['ALCOHOL_STATUS'] = df['ALQ101'].fillna(2).map({1: 1, 2: 0})
            composite_docs['ALCOHOL_STATUS'] = {
                'description': 'Lifetime alcohol use (0=no, 1=yes)'
            }
        
        print(f"\n=== Composite Feature Engineering ===")
        print(f"Created {len(composite_docs)} composite features")
        for feat, doc in composite_docs.items():
            print(f"  - {feat}: {doc['description']}")
        
        return df_eng, composite_docs
    
    def calculate_derived_features(self, df):
        """
        Calculate pharmacokinetic-relevant derived features.
        
        Returns:
        --------
        pd.DataFrame : Dataset with derived features
        dict : Documentation of calculations
        """
        df_derived = df.copy()
        derived_docs = {}
        
        # 1. Estimated GFR (CKD-EPI equation)
        # Requires: serum creatinine, age, sex, race
        req_egfr = ['LBXSCRSI', 'RIDAGEEX', 'RIAGENDR', 'RIDRETH1']
        
        if all(c in df.columns for c in ['LBXSCRSI', 'RIDAGEEX', 'RIAGENDR']):
            scr = df['LBXSCRSI']  # Serum creatinine in umol/L
            age = df['RIDAGEEX'] / 12  # Convert months to years
            sex = df['RIAGENDR']  # 1=male, 2=female
            
            # Convert creatinine to mg/dL for CKD-EPI
            scr_mgdl = scr / 88.4
            
            # CKD-EPI formula
            kappa = np.where(sex == 2, 0.7, 0.9)
            alpha = np.where(sex == 2, -0.329, -0.411)
            min_term = np.minimum(scr_mgdl / kappa, 1) ** alpha
            max_term = np.maximum(scr_mgdl / kappa, 1) ** (-1.209)
            sex_factor = np.where(sex == 2, 1.018, 1)
            age_factor = 0.993 ** age
            
            egfr = 141 * min_term * max_term * sex_factor * age_factor
            
            # Adjust for race if available
            if 'RIDRETH1' in df.columns:
                # RIDRETH1: 3=Non-Hispanic Black
                race_factor = np.where(df['RIDRETH1'] == 3, 1.159, 1)
                egfr = egfr * race_factor
            
            df_derived['EGFR_CKDEPI'] = egfr
            derived_docs['EGFR_CKDEPI'] = {
                'formula': 'CKD-EPI equation',
                'unit': 'mL/min/1.73m²',
                'reference': '≥90 normal'
            }
        
        # 2. Body Surface Area (Mosteller formula)
        if all(c in df.columns for c in ['BMXHT', 'BMXWT']):
            height_cm = df['BMXHT']
            weight_kg = df['BMXWT']
            
            bsa = np.sqrt((height_cm * weight_kg) / 3600)
            df_derived['BSA_MOSTELLER'] = bsa
            
            derived_docs['BSA_MOSTELLER'] = {
                'formula': 'sqrt((height_cm × weight_kg) / 3600)',
                'unit': 'm²',
                'reference': '1.7-2.0 typical adult'
            }
        
        # 3. Age groups for pharmacokinetic considerations
        if 'RIDAGEEX' in df.columns:
            age_years = df['RIDAGEEX'] / 12
            df_derived['AGE_GROUP'] = pd.cut(
                age_years,
                bins=[0, 18, 65, 150],
                labels=['pediatric', 'adult', 'elderly'],
                include_lowest=True
            )
            
            derived_docs['AGE_GROUP'] = {
                'categories': 'pediatric (<18), adult (18-65), elderly (>65)',
                'rationale': 'Age-dependent drug metabolism'
            }
        
        # 4. BMI category
        if 'BMXBMI' in df.columns:
            bmi = df['BMXBMI']
            df_derived['BMI_CATEGORY'] = pd.cut(
                bmi,
                bins=[0, 18.5, 25, 30, 100],
                labels=['underweight', 'normal', 'overweight', 'obese'],
                include_lowest=True
            )
            
            derived_docs['BMI_CATEGORY'] = {
                'categories': 'WHO BMI classification',
                'rationale': 'Body composition affects drug distribution'
            }
        
        # 5. Pregnancy indicator
        if 'RIDPREG' in df.columns:
            df_derived['PREGNANT'] = (df['RIDPREG'] == 1).astype(int)
            derived_docs['PREGNANT'] = {
                'values': '0=no, 1=yes',
                'rationale': 'Altered pharmacokinetics in pregnancy'
            }
        
        print(f"\n=== Derived Feature Calculation ===")
        print(f"Calculated {len(derived_docs)} derived features")
        for feat, doc in derived_docs.items():
            print(f"  - {feat}: {doc.get('formula', doc.get('categories', ''))}")
        
        return df_derived, derived_docs
    
    def encode_categorical_features(self, df):
        """
        Encode categorical variables for modeling.
        
        Returns:
        --------
        pd.DataFrame : Dataset with encoded features
        dict : Encoding mappings
        """
        df_encoded = df.copy()
        encoding_map = {}
        
        # One-hot encode race/ethnicity
        if 'RIDRETH1' in df.columns:
            race_dummies = pd.get_dummies(
                df['RIDRETH1'], 
                prefix='RACE',
                drop_first=True  # Avoid multicollinearity
            )
            df_encoded = pd.concat([df_encoded, race_dummies], axis=1)
            encoding_map['RIDRETH1'] = 'one-hot encoded to RACE_*'
        
        # Ordinal encode education
        if 'DMDEDUC' in df.columns:
            education_map = {
                1: 0,  # Less than 9th grade
                2: 1,  # 9-11th grade
                3: 2,  # High school grad
                4: 3,  # Some college
                5: 4   # College graduate or above
            }
            df_encoded['EDUCATION_ORDINAL'] = df['DMDEDUC'].map(education_map)
            encoding_map['DMDEDUC'] = education_map
        
        # Binary encode gender (if not already numeric)
        if 'RIAGENDR' in df.columns:
            df_encoded['GENDER_MALE'] = (df['RIAGENDR'] == 1).astype(int)
            encoding_map['RIAGENDR'] = {1: 'male', 2: 'female'}
        
        # Encode marital status
        if 'DMDMARTL' in df.columns:
            marital_dummies = pd.get_dummies(
                df['DMDMARTL'],
                prefix='MARITAL',
                drop_first=True
            )
            df_encoded = pd.concat([df_encoded, marital_dummies], axis=1)
            encoding_map['DMDMARTL'] = 'one-hot encoded to MARITAL_*'
        
        print(f"\n=== Categorical Encoding ===")
        print(f"Encoded {len(encoding_map)} categorical features")
        
        return df_encoded, encoding_map
    
    # ========================================================================
    # PHASE 4: MISSING DATA HANDLING
    # ========================================================================
    
    def analyze_missingness(self, df):
        """
        Comprehensive analysis of missing data patterns.
        
        Returns:
        --------
        pd.DataFrame : Missingness statistics
        """
        missing_stats = pd.DataFrame({
            'feature': df.columns,
            'n_missing': df.isnull().sum(),
            'pct_missing': (df.isnull().sum() / len(df)) * 100,
            'dtype': df.dtypes
        }).sort_values('pct_missing', ascending=False)
        
        print(f"\n=== Missingness Analysis ===")
        print(f"Features with >30% missing: {(missing_stats['pct_missing'] > 30).sum()}")
        print(f"Features with >50% missing: {(missing_stats['pct_missing'] > 50).sum()}")
        print("\nTop 10 features by missingness:")
        print(missing_stats.head(10)[['feature', 'pct_missing']])
        
        return missing_stats
    
    def create_missingness_indicators(self, df, threshold=0.20):
        """
        Create binary indicators for features with substantial missingness.
        
        Parameters:
        -----------
        threshold : float
            Minimum proportion missing to create indicator
            
        Returns:
        --------
        pd.DataFrame : Dataset with missingness indicators
        list : Names of created indicator features
        """
        df_with_indicators = df.copy()
        indicators_created = []
        
        for col in df.columns:
            pct_missing = df[col].isnull().sum() / len(df)
            
            if pct_missing >= threshold:
                indicator_name = f"{col}_MISSING"
                df_with_indicators[indicator_name] = df[col].isnull().astype(int)
                indicators_created.append(indicator_name)
        
        print(f"\n=== Missingness Indicator Creation ===")
        print(f"Created {len(indicators_created)} missingness indicators")
        
        return df_with_indicators, indicators_created
    
    # ============================================================================
    # REPLACE THIS FUNCTION IN THE NHANESPreprocessor CLASS
    # ============================================================================

    def impute_missing_values(self, df, method='hybrid'):
        """
        REVISED: Efficient imputation strategy that handles high-missingness gracefully.
        
        Parameters:
        -----------
        method : str
            Imputation method ('hybrid', 'simple', 'knn_selective')
            
        Returns:
        --------
        pd.DataFrame : Dataset with imputed values
        dict : Imputation metadata
        """
        df_imputed = df.copy()
        imputation_meta = {'method': method, 'features_imputed': {}}
        
        # Separate features by missingness level
        missing_pcts = (df.isnull().sum() / len(df)) * 100
        
        high_missing = missing_pcts[missing_pcts > 50].index.tolist()  # >50% missing
        medium_missing = missing_pcts[(missing_pcts > 10) & (missing_pcts <= 50)].index.tolist()
        low_missing = missing_pcts[(missing_pcts > 0) & (missing_pcts <= 10)].index.tolist()
        
        print(f"\n=== Efficient Missing Value Imputation ===")
        print(f"High missingness (>50%): {len(high_missing)} features")
        print(f"Medium missingness (10-50%): {len(medium_missing)} features")
        print(f"Low missingness (<10%): {len(low_missing)} features")
        
        # Separate numeric and categorical
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Strategy 1: Drop features with >90% missingness
        very_high_missing = missing_pcts[missing_pcts > 90].index.tolist()
        if very_high_missing:
            print(f"\n⚠️  Dropping {len(very_high_missing)} features with >90% missingness")
            df_imputed = df_imputed.drop(columns=very_high_missing)
            imputation_meta['dropped_high_missingness'] = very_high_missing
        
        # Update feature lists after dropping
        numeric_cols = [c for c in numeric_cols if c in df_imputed.columns]
        categorical_cols = [c for c in categorical_cols if c in df_imputed.columns]
        
        # Strategy 2: Simple imputation for high missingness (50-90%)
        high_missing_remaining = [c for c in high_missing if c in df_imputed.columns]
        
        for col in high_missing_remaining:
            if col in numeric_cols:
                median_val = df_imputed[col].median()
                df_imputed[col].fillna(median_val, inplace=True)
                imputation_meta['features_imputed'][col] = 'median'
            elif col in categorical_cols:
                mode_val = df_imputed[col].mode()[0] if not df_imputed[col].mode().empty else 'Unknown'
                df_imputed[col].fillna(mode_val, inplace=True)
                imputation_meta['features_imputed'][col] = 'mode'
        
        print(f"Applied simple imputation to {len(high_missing_remaining)} high-missingness features")
        
        # Strategy 3: KNN imputation for medium missingness (batched)
        medium_missing_numeric = [c for c in medium_missing if c in numeric_cols and c in df_imputed.columns]
        
        if medium_missing_numeric:
            # Process in batches to avoid memory issues
            batch_size = 50
            n_batches = (len(medium_missing_numeric) + batch_size - 1) // batch_size
            
            print(f"\nApplying KNN imputation to {len(medium_missing_numeric)} medium-missingness features")
            print(f"Processing in {n_batches} batches of {batch_size}...")
            
            from sklearn.impute import KNNImputer
            knn_imputer = KNNImputer(n_neighbors=5)
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(medium_missing_numeric))
                batch_cols = medium_missing_numeric[start_idx:end_idx]
                
                print(f"  Batch {i+1}/{n_batches}: {len(batch_cols)} features", end='')
                
                # Only impute this batch
                df_imputed[batch_cols] = knn_imputer.fit_transform(df_imputed[batch_cols])
                
                for col in batch_cols:
                    imputation_meta['features_imputed'][col] = 'knn'
                
                print(" ✓")
        
        # Strategy 4: Simple imputation for low missingness
        low_missing_remaining = [c for c in low_missing if c in df_imputed.columns]
        
        for col in low_missing_remaining:
            if df_imputed[col].isnull().any():
                if col in numeric_cols:
                    median_val = df_imputed[col].median()
                    df_imputed[col].fillna(median_val, inplace=True)
                    imputation_meta['features_imputed'][col] = 'median'
                elif col in categorical_cols:
                    mode_val = df_imputed[col].mode()[0] if not df_imputed[col].mode().empty else 'Unknown'
                    df_imputed[col].fillna(mode_val, inplace=True)
                    imputation_meta['features_imputed'][col] = 'mode'
        
        print(f"Applied simple imputation to {len(low_missing_remaining)} low-missingness features")
        
        # Final check
        remaining_nulls = df_imputed.isnull().sum().sum()
        print(f"\n✅ Imputation complete. Remaining null values: {remaining_nulls}")
        
        if remaining_nulls > 0:
            print("⚠️  Some null values remain. Filling with 0 (fallback)...")
            df_imputed = df_imputed.fillna(0)
        
        return df_imputed, imputation_meta

    
    # ========================================================================
    # PHASE 5: NORMALIZATION AND SCALING
    # ========================================================================
    
    def apply_transformations(self, df):
        """
        Apply appropriate transformations to skewed distributions.
        
        Returns:
        --------
        pd.DataFrame : Transformed dataset
        dict : Transformation metadata
        """
        df_transformed = df.copy()
        transform_meta = {}
        
        # Identify highly skewed features (log transformation candidates)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            skewness = df[col].skew()
            
            # Apply log transformation if right-skewed and positive
            if skewness > 1.5 and (df[col] > 0).all():
                df_transformed[f"{col}_LOG"] = np.log1p(df[col])  # log1p handles zeros
                transform_meta[col] = {
                    'transformation': 'log1p',
                    'original_skewness': skewness,
                    'new_feature': f"{col}_LOG"
                }
        
        print(f"\n=== Distribution Transformations ===")
        print(f"Applied log transformation to {len(transform_meta)} features")
        
        return df_transformed, transform_meta
    
    def scale_features(self, df, method='standard'):
        """
        Scale numeric features for modeling.
        
        Parameters:
        -----------
        method : str
            Scaling method ('standard', 'robust', 'minmax')
            
        Returns:
        --------
        pd.DataFrame : Scaled dataset
        sklearn scaler object : Fitted scaler
        """
        df_scaled = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform numeric columns
        df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        print(f"\n=== Feature Scaling ({method}) ===")
        print(f"Scaled {len(numeric_cols)} numeric features")
        print(f"Feature ranges after scaling: [{df_scaled[numeric_cols].min().min():.2f}, "
              f"{df_scaled[numeric_cols].max().max():.2f}]")
        
        return df_scaled, scaler
    
    # ========================================================================
    # PHASE 6: OUTLIER DETECTION
    # ========================================================================
    
    def detect_outliers(self, df, method='iqr', threshold=1.5):
        """
        Detect outliers using statistical methods.
        
        Parameters:
        -----------
        method : str
            Detection method ('iqr', 'zscore')
        threshold : float
            IQR multiplier or z-score threshold
            
        Returns:
        --------
        pd.DataFrame : Boolean DataFrame indicating outliers
        dict : Outlier statistics
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
        outlier_stats = {}
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask[col] = z_scores > threshold
            
            outlier_count = outlier_mask[col].sum()
            outlier_stats[col] = {
                'n_outliers': outlier_count,
                'pct_outliers': (outlier_count / len(df)) * 100
            }
        
        total_outliers = outlier_mask.sum().sum()
        print(f"\n=== Outlier Detection ({method}, threshold={threshold}) ===")
        print(f"Detected {total_outliers} outlier values across {len(numeric_cols)} features")
        print(f"Features with >5% outliers: "
              f"{sum(1 for v in outlier_stats.values() if v['pct_outliers'] > 5)}")
        
        return outlier_mask, outlier_stats
    
    def flag_clinical_implausibility(self, df):
        """
        Flag physiologically impossible values based on clinical knowledge.
        
        Returns:
        --------
        pd.DataFrame : Dataset with implausible values set to NaN
        dict : Count of flagged values per feature
        """
        df_clean = df.copy()
        flagged_counts = {}
        
        # Define clinical plausibility ranges (non-exhaustive examples)
        clinical_ranges = {
            'LBXHGB': (3, 20),          # Hemoglobin: 3-20 g/dL
            'LBXGLUSI': (1.1, 33.3),    # Glucose: 20-600 mg/dL → 1.1-33.3 mmol/L
            'LBXWBCSI': (1, 100),       # WBC: 1-100 K/uL
            'LBXPLTSI': (10, 1000),     # Platelets: 10-1000 K/uL
            'LBXSCRSI': (10, 1500),     # Creatinine: 0.1-17 mg/dL → 10-1500 umol/L
            'BMXBMI': (10, 100),        # BMI: 10-100 kg/m²
            'LBXTCSI': (1, 20),         # Total cholesterol: 40-800 mg/dL → 1-20 mmol/L
        }
        
        for feature, (min_val, max_val) in clinical_ranges.items():
            if feature in df.columns:
                outliers = (df[feature] < min_val) | (df[feature] > max_val)
                n_flagged = outliers.sum()
                
                if n_flagged > 0:
                    df_clean.loc[outliers, feature] = np.nan
                    flagged_counts[feature] = n_flagged
        
        print(f"\n=== Clinical Plausibility Flagging ===")
        print(f"Flagged {sum(flagged_counts.values())} implausible values across "
              f"{len(flagged_counts)} features")
        
        return df_clean, flagged_counts
    
    # ========================================================================
    # PHASE 7: FEATURE SELECTION
    # ========================================================================
    
    def select_by_variance(self, df):
        """
        Remove near-zero variance features.
        
        Returns:
        --------
        pd.DataFrame : Dataset with low-variance features removed
        list : Dropped feature names
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        low_var_features = []
        for col in numeric_cols:
            var = df[col].var()
            if var < self.variance_threshold:
                low_var_features.append(col)
        
        df_filtered = df.drop(columns=low_var_features)
        
        print(f"\n=== Variance Filtering ===")
        print(f"Dropped {len(low_var_features)} near-zero variance features")
        
        return df_filtered, low_var_features
    
    def prioritize_by_domain_knowledge(self, df):
        """
        Rank features by clinical importance for blood-drug modeling.
        
        Returns:
        --------
        dict : Three-tiered feature priority ranking
        """
        priority_tiers = {
            'tier1_critical': [],
            'tier2_important': [],
            'tier3_supplementary': []
        }
        
        # Tier 1: Critical features
        tier1_patterns = [
            'RIDAGEEX',        # Age
            'RIAGENDR',        # Sex
            'BMXBMI', 'BMXWT', # Body composition
            'LBXWBCSI', 'LBXRBCSI', 'LBXHGB', 'LBXHCT', 'LBXPLTSI',  # CBC
            'LBXSGTSI', 'LBXSATSI', 'LBXSTBSI', 'LBXSALSI',  # Liver function
            'LBXSCRSI', 'EGFR',  # Kidney function
            'LBXGLUSI',         # Glucose
            'LBXTCSI', 'LBXHDDSI', 'LBXLDLSI', 'LBXTRSI',  # Lipids
        ]
        
        # Tier 2: Important features
        tier2_patterns = [
            'LBXSKSI', 'LBXSNASI', 'LBXSCLSI', 'LBXSCASI',  # Electrolytes
            'LBXTSH', 'LBXT4',    # Thyroid
            'RIDPREG', 'PREGNANT', # Pregnancy
            'CVD_COMPOSITE', 'DIABETES_COMPOSITE', 'LIVER_DISEASE', 'KIDNEY_DISEASE',  # Comorbidities
            'RXDDRGID',          # Medications
        ]
        
        # Tier 3: Supplementary features
        tier3_patterns = [
            'LBXVIDLC', 'LBXVD2MS', 'LBXVD3MS',  # Vitamins
            'LBXFERSI', 'LBXIRNSI',  # Iron
            'LBXBPBSI', 'LBXBCDSI', 'LBXTHGSI',  # Heavy metals
            'LBXCRP',              # Inflammation
            'LBXINSI',             # Insulin
            'LBXFSH', 'LBXLH',     # Other hormones
        ]
        
        # Classify available features
        for col in df.columns:
            if any(pattern in col for pattern in tier1_patterns):
                priority_tiers['tier1_critical'].append(col)
            elif any(pattern in col for pattern in tier2_patterns):
                priority_tiers['tier2_important'].append(col)
            elif any(pattern in col for pattern in tier3_patterns):
                priority_tiers['tier3_supplementary'].append(col)
        
        print(f"\n=== Domain Knowledge Feature Prioritization ===")
        print(f"Tier 1 (Critical): {len(priority_tiers['tier1_critical'])} features")
        print(f"Tier 2 (Important): {len(priority_tiers['tier2_important'])} features")
        print(f"Tier 3 (Supplementary): {len(priority_tiers['tier3_supplementary'])} features")
        
        return priority_tiers
    
    # ========================================================================
    # PHASE 8: DATA QUALITY AND EXPORT
    # ========================================================================
    
    def run_quality_checks(self, df):
        """
        Perform comprehensive data quality checks.
        
        Returns"""
        quality_report = {
            'checks_passed': [],
            'checks_failed': [],
            'warnings': []
        }
        
        # Check 1: Pregnancy status consistency
        if all(col in df.columns for col in ['PREGNANT', 'RIAGENDR']):
            male_pregnant = df[(df['RIAGENDR'] == 1) & (df['PREGNANT'] == 1)]
            if len(male_pregnant) > 0:
                quality_report['checks_failed'].append({
                    'check': 'Pregnancy-Gender Consistency',
                    'issue': f'{len(male_pregnant)} males marked as pregnant',
                    'action': 'Set PREGNANT=0 for males'
                })
                df.loc[(df['RIAGENDR'] == 1), 'PREGNANT'] = 0
            else:
                quality_report['checks_passed'].append('Pregnancy-Gender Consistency')
        
        # Check 2: Age-dependent features
        if 'RIDAGEEX' in df.columns:
            age_years = df['RIDAGEEX'] / 12
            
            # Check for pediatric subjects with adult-only measurements
            pediatric_mask = age_years < 18
            if pediatric_mask.sum() > 0:
                quality_report['warnings'].append({
                    'warning': 'Pediatric subjects present',
                    'count': pediatric_mask.sum(),
                    'recommendation': 'Consider age-specific reference ranges'
                })
        
        # Check 3: Value ranges for critical features
        range_checks = {
            'LBXHGB': (3, 20, 'Hemoglobin'),
            'LBXWBCSI': (1, 100, 'White blood cell count'),
            'BMXBMI': (10, 100, 'Body mass index')
        }
        
        for feature, (min_val, max_val, name) in range_checks.items():
            if feature in df.columns:
                out_of_range = ((df[feature] < min_val) | (df[feature] > max_val)).sum()
                if out_of_range == 0:
                    quality_report['checks_passed'].append(f'{name} within expected range')
                else:
                    quality_report['warnings'].append({
                        'warning': f'{name} has {out_of_range} values outside expected range',
                        'feature': feature,
                        'expected_range': f'{min_val}-{max_val}'
                    })
        
        # Check 4: Missing data completeness
        missing_pct = (df.isnull().sum() / len(df) * 100).max()
        if missing_pct < 10:
            quality_report['checks_passed'].append('Low missingness (<10% max per feature)')
        elif missing_pct < 30:
            quality_report['warnings'].append({
                'warning': f'Moderate missingness detected (max {missing_pct:.1f}%)',
                'recommendation': 'Review imputation strategy'
            })
        else:
            quality_report['checks_failed'].append({
                'check': 'Missing Data Threshold',
                'issue': f'High missingness (max {missing_pct:.1f}%)',
                'action': 'Consider removing high-missingness features'
            })
        
        # Check 5: Duplicate rows
        n_duplicates = df.duplicated().sum()
        if n_duplicates == 0:
            quality_report['checks_passed'].append('No duplicate rows')
        else:
            quality_report['checks_failed'].append({
                'check': 'Duplicate Rows',
                'issue': f'{n_duplicates} duplicate rows found',
                'action': 'Remove duplicates'
            })
            df = df.drop_duplicates()
        
        # Check 6: Data type consistency
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].dtype == 'object':
                quality_report['checks_failed'].append({
                    'check': 'Data Type Consistency',
                    'issue': f'{col} is numeric but stored as object',
                    'action': 'Convert to numeric'
                })
        
        print(f"\n=== Data Quality Report ===")
        print(f"Checks Passed: {len(quality_report['checks_passed'])}")
        print(f"Checks Failed: {len(quality_report['checks_failed'])}")
        print(f"Warnings: {len(quality_report['warnings'])}")
        
        if quality_report['checks_failed']:
            print("\nFailed Checks:")
            for fail in quality_report['checks_failed']:
                print(f"  ❌ {fail['check']}: {fail['issue']}")
        
        if quality_report['warnings']:
            print("\nWarnings:")
            for warn in quality_report['warnings']:
                print(f"  ⚠️  {warn['warning']}")
        
        return df, quality_report
    
    def create_feature_documentation(self, df, all_metadata):
        """
        Generate comprehensive feature documentation.
        
        Parameters:
        -----------
        all_metadata : dict
            Combined metadata from all preprocessing steps
            
        Returns:
        --------
        pd.DataFrame : Feature documentation table
        """
        doc_data = []
        
        for col in df.columns:
            doc_entry = {
                'feature_name': col,
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'coverage_pct': (df[col].count() / len(df)) * 100,
                'unique_values': df[col].nunique(),
            }
            
            # Add statistics for numeric features
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                doc_entry.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'q25': df[col].quantile(0.25),
                    'median': df[col].median(),
                    'q75': df[col].quantile(0.75),
                    'max': df[col].max(),
                    'skewness': df[col].skew(),
                })
            
            # Add transformation info if available
            if 'transformations' in all_metadata:
                if col in all_metadata['transformations']:
                    doc_entry['transformation'] = all_metadata['transformations'][col]['transformation']
            
            # Add composite feature info if available
            if 'composite_features' in all_metadata:
                if col in all_metadata['composite_features']:
                    doc_entry['description'] = all_metadata['composite_features'][col]['description']
            
            # Add derived feature info if available
            if 'derived_features' in all_metadata:
                if col in all_metadata['derived_features']:
                    doc_entry['formula'] = all_metadata['derived_features'][col].get('formula', '')
                    doc_entry['unit'] = all_metadata['derived_features'][col].get('unit', '')
            
            doc_data.append(doc_entry)
        
        feature_doc = pd.DataFrame(doc_data)
        
        print(f"\n=== Feature Documentation Generated ===")
        print(f"Documented {len(feature_doc)} features")
        print(f"Documentation columns: {', '.join(feature_doc.columns)}")
        
        return feature_doc
    
    def export_preprocessed_data(self, df, feature_doc, output_dir='./preprocessed_nhanes/'):
        """
        Export preprocessed data and documentation.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Preprocessed dataset
        feature_doc : pd.DataFrame
            Feature documentation
        output_dir : str
            Output directory path
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export main dataset
        df.to_csv(f'{output_dir}nhanes_preprocessed.csv', index=False)
        print(f"\n✅ Exported preprocessed data: {output_dir}nhanes_preprocessed.csv")
        print(f"   Shape: {df.shape}")
        print(f"   Size: {os.path.getsize(f'{output_dir}nhanes_preprocessed.csv') / 1e6:.2f} MB")
        
        # Export feature documentation
        feature_doc.to_csv(f'{output_dir}feature_documentation.csv', index=False)
        print(f"✅ Exported feature documentation: {output_dir}feature_documentation.csv")
        
        # Export train-test split indices for reproducibility
        from sklearn.model_selection import train_test_split
        
        train_idx, test_idx = train_test_split(
            df.index, 
            test_size=0.2, 
            random_state=42,
            stratify=df['RIAGENDR'] if 'RIAGENDR' in df.columns else None
        )
        
        np.save(f'{output_dir}train_indices.npy', train_idx)
        np.save(f'{output_dir}test_indices.npy', test_idx)
        print(f"✅ Exported train/test split indices")
        print(f"   Train: {len(train_idx)} samples ({len(train_idx)/len(df)*100:.1f}%)")
        print(f"   Test: {len(test_idx)} samples ({len(test_idx)/len(df)*100:.1f}%)")
        
        return True


def main_preprocessing_pipeline(input_file, output_dir='./preprocessed_nhanes/'):
    """
    REVISED: Optimized preprocessing pipeline that doesn't hang on large datasets.
    
    Parameters:
    -----------
    input_file : str
        Path to raw NHANES CSV file
    output_dir : str
        Output directory for preprocessed data
        
    Returns:
    --------
    pd.DataFrame : Preprocessed dataset
    dict : Complete preprocessing metadata
    pd.DataFrame : Feature documentation
    """
    
    print("="*80)
    print("NHANES DATA PREPROCESSING PIPELINE")
    print("For Blood-Drug Interaction Simulation Modeling")
    print("="*80)
    
    # Initialize preprocessor
    preprocessor = NHANESPreprocessor(
        coverage_threshold=0.05,  # 5% minimum coverage
        variance_threshold=0.01
    )
    
    # Track all metadata
    metadata = {}
    
    # ========================================================================
    # PHASE 1: LOAD AND FILTER
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 1: DATA LOADING AND INITIAL FILTERING")
    print("="*80)
    
    df = preprocessor.load_data(input_file)
    
    # Analyze coverage
    coverage_stats = preprocessor.analyze_coverage(df)
    metadata['initial_coverage'] = coverage_stats
    
    # REVISED: More aggressive initial filtering to reduce computational burden
    print("\n🔧 Applying aggressive filtering for computational efficiency...")
    
    # Drop features with >80% missingness immediately
    high_missing_features = coverage_stats[coverage_stats['coverage_pct'] < 20]['feature'].tolist()
    print(f"Dropping {len(high_missing_features)} features with >80% missingness")
    df = df.drop(columns=high_missing_features)
    metadata['dropped_high_missingness_initial'] = high_missing_features

    # Recompute coverage statistics after the aggressive drop so subsequent
    # filtering uses a current view of existing columns
    coverage_stats = preprocessor.analyze_coverage(df)

    # Optionally drop rows with excessive missingness early
    df, dropped_sparse_rows = preprocessor.drop_sparse_rows(df)
    metadata['dropped_sparse_rows'] = dropped_sparse_rows

    # Filter by coverage (safe selection inside method)
    df, dropped_coverage = preprocessor.filter_by_coverage(df, coverage_stats)
    metadata['dropped_coverage'] = dropped_coverage
    
    # Drop irrelevant categories
    df, dropped_irrelevant = preprocessor.drop_irrelevant_categories(df)
    metadata['dropped_irrelevant'] = dropped_irrelevant
    
    # ========================================================================
    # PHASE 2: HANDLE REDUNDANCY
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: REDUNDANCY RESOLUTION")
    print("="*80)
    
    # Consolidate unit duplicates
    df, unit_map = preprocessor.consolidate_unit_duplicates(df)
    metadata['unit_consolidation'] = unit_map
    
    # Consolidate longitudinal measurements
    df, consol_map = preprocessor.consolidate_longitudinal_labs(df)
    metadata['longitudinal_consolidation'] = consol_map

    # Consolidate features by prefix to merge cycle-specific variants
    df, prefix_map = preprocessor.consolidate_by_prefix(df)
    metadata['prefix_consolidation'] = prefix_map
    
    # Remove high correlation
    df, dropped_corr = preprocessor.remove_high_correlation(df, threshold=0.95)
    metadata['dropped_correlation'] = dropped_corr
    
    # ========================================================================
    # PHASE 3: FEATURE ENGINEERING
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 3: FEATURE ENGINEERING")
    print("="*80)
    
    # Create composite features
    df, composite_docs = preprocessor.create_composite_features(df)
    metadata['composite_features'] = composite_docs
    
    # Calculate derived features
    df, derived_docs = preprocessor.calculate_derived_features(df)
    metadata['derived_features'] = derived_docs
    
    # Encode categorical features
    df, encoding_map = preprocessor.encode_categorical_features(df)
    metadata['encoding_map'] = encoding_map
    
    # ========================================================================
    # PHASE 4: MISSING DATA (REVISED - EFFICIENT)
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 4: EFFICIENT MISSING DATA HANDLING")
    print("="*80)
    
    # Analyze missingness
    missing_stats = preprocessor.analyze_missingness(df)
    metadata['missingness_analysis'] = missing_stats
    
    # REVISED: Only create missingness indicators for medium-missingness features
    print("\n🔧 Creating missingness indicators only for medium-missingness features (20-50%)...")
    missing_pcts = (df.isnull().sum() / len(df)) * 100
    medium_missing_features = missing_pcts[(missing_pcts >= 20) & (missing_pcts <= 50)].index.tolist()
    
    missing_indicators = []
    for col in medium_missing_features:
        indicator_name = f"{col}_MISSING"
        df[indicator_name] = df[col].isnull().astype(int)
        missing_indicators.append(indicator_name)
    
    print(f"Created {len(missing_indicators)} missingness indicators")
    metadata['missingness_indicators'] = missing_indicators
    
    # REVISED: Use efficient imputation (this calls the new impute_missing_values)
    df, imputation_meta = preprocessor.impute_missing_values(df, method='hybrid')
    metadata['imputation'] = imputation_meta
    
    # ========================================================================
    # PHASE 5: TRANSFORMATIONS AND SCALING
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 5: TRANSFORMATIONS AND SCALING")
    print("="*80)
    
    # Apply transformations
    df, transform_meta = preprocessor.apply_transformations(df)
    metadata['transformations'] = transform_meta
    
    # Note: Scaling should be done after train-test split to avoid leakage
    print("\n⚠️  Note: Final scaling should be applied after train-test split")
    print("   to prevent data leakage. Scaler will be fitted on training data only.")
    
    # ========================================================================
    # PHASE 6: OUTLIER HANDLING
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 6: OUTLIER DETECTION AND HANDLING")
    print("="*80)
    
    # Flag clinical implausibility
    df, flagged_counts = preprocessor.flag_clinical_implausibility(df)
    metadata['flagged_implausible'] = flagged_counts
    
    # REVISED: Skip computationally intensive outlier detection on full dataset
    print("⚠️  Skipping full outlier detection (can be done on modeling subset if needed)")
    metadata['outlier_detection'] = "Skipped for computational efficiency"
    
    # ========================================================================
    # PHASE 7: FEATURE SELECTION
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 7: FEATURE SELECTION")
    print("="*80)
    
    # Variance filtering
    df, low_var_features = preprocessor.select_by_variance(df)
    metadata['dropped_low_variance'] = low_var_features
    
    # Domain knowledge prioritization
    priority_tiers = preprocessor.prioritize_by_domain_knowledge(df)
    metadata['feature_priorities'] = priority_tiers
    
    # ========================================================================
    # PHASE 8: QUALITY ASSURANCE
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 8: DATA QUALITY ASSURANCE")
    print("="*80)
    
    # Run quality checks
    df, quality_report = preprocessor.run_quality_checks(df)
    metadata['quality_report'] = quality_report
    
    # Create feature documentation
    feature_doc = preprocessor.create_feature_documentation(df, metadata)
    
    # ========================================================================
    # EXPORT
    # ========================================================================
    print("\n" + "="*80)
    print("EXPORTING PREPROCESSED DATA")
    print("="*80)
    
    preprocessor.export_preprocessed_data(df, feature_doc, output_dir)
    
    # Export metadata
    import json
    
    # Convert non-serializable objects to strings
    metadata_serializable = {}
    for key, value in metadata.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            metadata_serializable[key] = value.to_dict()
        elif isinstance(value, dict):
            metadata_serializable[key] = {str(k): str(v) for k, v in value.items()}
        else:
            metadata_serializable[key] = str(value)
    
    with open(f'{output_dir}preprocessing_metadata.json', 'w') as f:
        # Use default=str to ensure non-serializable objects (pandas dtypes,
        # numpy types) are converted to strings rather than causing a crash.
        json.dump(metadata_serializable, f, indent=2, default=str)
    
    print(f"✅ Exported preprocessing metadata: {output_dir}preprocessing_metadata.json")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE - SUMMARY")
    print("="*80)
    print(f"Final dataset shape: {df.shape}")
    print(f"Features retained: {df.shape[1]}")
    print(f"Observations: {df.shape[0]}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"\nFeature breakdown:")
    print(f"  - Tier 1 (Critical): {len(priority_tiers['tier1_critical'])}")
    print(f"  - Tier 2 (Important): {len(priority_tiers['tier2_important'])}")
    print(f"  - Tier 3 (Supplementary): {len(priority_tiers['tier3_supplementary'])}")
    print(f"\nData quality:")
    print(f"  - Missing values: {df.isnull().sum().sum()} ({df.isnull().sum().sum() / df.size * 100:.2f}%)")
    print(f"  - Duplicate rows: {df.duplicated().sum()}")
    print(f"\nNext steps:")
    print("  1. Review feature_documentation.csv for detailed feature information")
    print("  2. Perform train-test split using provided indices")
    print("  3. Apply scaling to training set and transform test set")
    print("  4. Proceed to modeling phase")
    print("="*80)
    
    return df, metadata, feature_doc


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_train_test_split(df, test_size=0.2, stratify_col=None, random_state=42):
    """
    Create stratified train-test split for modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataset
    test_size : float
        Proportion of data for testing
    stratify_col : str
        Column name to stratify by (e.g., 'RIAGENDR')
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    print(f"Train-Test Split:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    if stratify_col and stratify_col in df.columns:
        print(f"\nStratification by {stratify_col}:")
        print("Train distribution:")
        print(train_df[stratify_col].value_counts(normalize=True))
        print("\nTest distribution:")
        print(test_df[stratify_col].value_counts(normalize=True))
    
    return train_df, test_df


def apply_scaling_to_splits(train_df, test_df, method='standard'):
    """
    Apply scaling to train and test sets without data leakage.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training set
    test_df : pd.DataFrame
        Test set
    method : str
        Scaling method ('standard' or 'robust')
        
    Returns:
    --------
    tuple : (train_scaled, test_scaled, scaler)
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler
    
    # Select numeric columns, but exclude ID columns (SEQN, etc.)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    id_cols = ['SEQN']  # Columns to preserve without scaling
    numeric_cols = [c for c in numeric_cols if c not in id_cols]
    
    # Initialize scaler
    scaler = StandardScaler() if method == 'standard' else RobustScaler()
    
    # Fit on training data only
    scaler.fit(train_df[numeric_cols])
    
    # Transform both sets
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    
    train_scaled[numeric_cols] = scaler.transform(train_df[numeric_cols])
    test_scaled[numeric_cols] = scaler.transform(test_df[numeric_cols])
    
    print(f"Applied {method} scaling to {len(numeric_cols)} features (excluding ID columns: {id_cols})")
    print(f"Scaler fitted on training data only (n={len(train_df)})")
    
    return train_scaled, test_scaled, scaler


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example execution of preprocessing pipeline.
    
    Note: This assumes you have a consolidated NHANES CSV file.
    Adjust file paths as needed.
    """
    
    # File paths (adjust as needed)
    INPUT_FILE = "./raw_data/Nhanes_complete.csv"
    OUTPUT_DIR = "./preprocessed_nhanes/"
    
    # Run complete preprocessing pipeline
    df_preprocessed, metadata, feature_doc = main_preprocessing_pipeline(
        input_file=INPUT_FILE,
        output_dir=OUTPUT_DIR
    )
    
    # Create train-test split
    train_df, test_df = create_train_test_split(
        df_preprocessed,
        test_size=0.2,
        stratify_col='RIAGENDR',
        random_state=42
    )
    
    # Apply scaling
    train_scaled, test_scaled, scaler = apply_scaling_to_splits(
        train_df, test_df, method='standard'
    )
    
    # Save splits
    train_scaled.to_csv(f'{OUTPUT_DIR}train_scaled.csv', index=False)
    test_scaled.to_csv(f'{OUTPUT_DIR}test_scaled.csv', index=False)
    
    print("\n✅ Preprocessing pipeline complete!")
    print(f"   All outputs saved to: {OUTPUT_DIR}")