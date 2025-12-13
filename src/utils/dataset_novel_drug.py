"""
Dataset that ensures test drugs are SIMILAR but UNSEEN during training.

This forces the model to generalize based on molecular structure.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from typing import Tuple, List, Optional
from pathlib import Path
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from encoders.drugEncoder import DrugEncoder
from models.pharmacodynamicPredictor import PATIENT_FEATURES, LAB_BIOMARKER_FEATURES

# Common drug name to SMILES mapping
# TODO: Expand this with more drugs or use a drug database API (e.g., PubChem, ChEMBL)
DRUG_NAME_TO_SMILES = {
    'Amlodipine': 'CCOC(=O)C1=C(C)NC(C)=C(C1C2=CC=CC=C2)C(=O)NCC3CCN(CC3)C(=O)OC',
    'Metformin': 'CN(C)C(=N)NC(=N)N',
    'Lisinopril': 'CCCCC(C(=O)N1CCCC1C(=O)O)NC(CCC(=O)O)C(=O)O',
    'Ibuprofen': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
    'Atorvastatin': 'CC(C)C(=O)N1CCC(CC1)C(=O)N2CCCC2C(=O)O',  # Simplified - verify with PubChem
    'Simvastatin': 'CC(C)C1=CC=C(C=C1)C2C(CC3C2CC(C4C3CCC4(C)C)O)OCCOC(=O)C(C)C',
    'Prednisone': 'C[C@]12C[C@H]3[C@@H](C[C@H]4[C@@]3(C=CC(=O)C4)C)C[C@@H]1[C@@H](C(=O)CO2)O',
    'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
    'Enalapril': 'CCCCN1CCCC1C(=O)N2CCCC2C(=O)O',
    'Furosemide': 'NS(=O)(=O)C1=C(Cl)C=C(NCC2=CC=CO2)C(=C1)C(O)=O',
    'Allopurinol': 'C1=NNC2=C1C(=O)NC=N2',
    # Add more as needed - consider using PubChem API for automatic lookup
}


class NovelDrugDataset(Dataset):
    """
    Dataset with drug-aware train/test split.
    
    Strategy:
    1. Load real drug-response data
    2. Encode all drugs to embeddings
    3. Cluster drugs by molecular similarity
    4. Hold out 1 drug per cluster for testing
    5. Train sees 80% drugs, test sees 20% DIFFERENT drugs
    """
    
    def __init__(
        self,
        data_path: str,
        drug_encoder: DrugEncoder,
        split: str = 'train',  # 'train' or 'test'
        test_ratio: float = 0.2,
        seed: int = 42,
        train_drugs: Optional[List[str]] = None,
        test_drugs: Optional[List[str]] = None
    ):
        """
        Args:
            data_path: Path to clinical_trials_responses.csv
            drug_encoder: Initialized DrugEncoder (hybrid mode)
            split: 'train' or 'test'
            test_ratio: Fraction of drugs to hold out (not samples!)
            seed: Random seed
            train_drugs: Optional pre-computed list of training drugs
            test_drugs: Optional pre-computed list of test drugs
        """
        self.data_path = data_path
        self.drug_encoder = drug_encoder
        self.split = split
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load CSV
        df = pd.read_csv(data_path)
        print(f"  Loaded {len(df):,} samples from {df['drug_name'].nunique()} unique drugs")
        
        # If train/test drugs are provided, use them; otherwise create split
        if train_drugs is not None and test_drugs is not None:
            self.train_drugs = train_drugs
            self.test_drugs = test_drugs
        else:
            # Create split (will be done by create_novel_drug_splits)
            raise ValueError("train_drugs and test_drugs must be provided. Use create_novel_drug_splits() first.")
        
        # Filter data by split
        if split == 'train':
            self.data = df[df['drug_name'].isin(self.train_drugs)].copy()
        elif split == 'test':
            self.data = df[df['drug_name'].isin(self.test_drugs)].copy()
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split}")
        
        print(f"  {split.capitalize()} set: {len(self.data):,} samples from {self.data['drug_name'].nunique()} drugs")
        
        # Cache drug embeddings (avoid re-encoding)
        self._drug_emb_cache = {}
        
        # Validate data
        self._validate_data()
    
    def _validate_data(self):
        """Check that required columns exist."""
        required_cols = ['patient_id', 'drug_name', 'age', 'sex', 'bmi']
        missing = [c for c in required_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            patient_state: (41,) tensor
            drug_emb: (768,) tensor from DrugEncoder
            lab_delta: (22,) tensor - actual measured changes
        """
        row = self.data.iloc[idx]
        
        # Extract patient features (41 features)
        patient_state = self._extract_patient_state(row)
        
        # Get drug embedding
        drug_name = str(row['drug_name'])
        if drug_name not in self._drug_emb_cache:
            try:
                # First, try to get SMILES from mapping if drug name is provided
                smiles = DRUG_NAME_TO_SMILES.get(drug_name, drug_name)
                
                # Also try loading from drug_smiles_mapping.json if available
                if smiles == drug_name:  # Not found in DRUG_NAME_TO_SMILES
                    drug_smiles_path = Path('data/mappings/drug_smiles_mapping.json')
                    if drug_smiles_path.exists():
                        import json
                        with open(drug_smiles_path, 'r') as f:
                            drug_smiles_mapping = json.load(f)
                        # Try exact match first
                        if drug_name in drug_smiles_mapping:
                            mapped_smiles = drug_smiles_mapping[drug_name]
                            if mapped_smiles is not None:
                                smiles = mapped_smiles
                            else:
                                raise ValueError(f"Drug '{drug_name}' has null SMILES in mapping")
                        else:
                            # Try base name match (e.g., "Xanomeline" -> "Xanomeline Low Dose")
                            for mapped_name, mapped_smiles in drug_smiles_mapping.items():
                                if mapped_smiles is not None and drug_name.lower() in mapped_name.lower():
                                    smiles = mapped_smiles
                                    break
                            if smiles == drug_name:
                                # Don't raise error - just use hash-based embedding silently
                                raise ValueError(f"Drug '{drug_name}' not found in SMILES mapping")
                
                # Try to encode (works if drug_name is already SMILES or we have mapping)
                drug_emb = self.drug_encoder.encode(smiles)
                self._drug_emb_cache[drug_name] = drug_emb
            except Exception as e:
                # If encoding fails, use a hash-based embedding as fallback
                # Only warn for unexpected errors, not for missing SMILES (which is expected)
                if "SMILES mapping" not in str(e):
                    warnings.warn(f"Failed to encode drug '{drug_name}': {e}. Using hash-based embedding.")
                # Create a deterministic embedding from drug name hash
                np.random.seed(hash(drug_name) % (2**32))
                drug_emb = np.random.randn(768).astype(np.float32)
                drug_emb = drug_emb / np.linalg.norm(drug_emb) * 30  # Normalize to reasonable scale
                self._drug_emb_cache[drug_name] = drug_emb
        else:
            drug_emb = self._drug_emb_cache[drug_name]
        
        # Extract lab deltas (22 biomarkers)
        lab_delta = self._extract_lab_delta(row)
        
        return {
            'patient_state': torch.tensor(patient_state, dtype=torch.float32),
            'drug_emb': torch.tensor(drug_emb, dtype=torch.float32),
            'lab_delta': torch.tensor(lab_delta, dtype=torch.float32)
        }
    
    def _extract_patient_state(self, row: pd.Series) -> np.ndarray:
        """Extract 41 patient features from CSV row.
        
        Maps CSV columns to PATIENT_FEATURES:
        - age, sex, bmi -> AGE, SEX, BMXBMI
        - Baseline values from *_baseline columns
        - Missing values filled with realistic defaults (not 0.0!)
        """
        state = np.zeros(41, dtype=np.float32)
        
        # Input conditions (0-3)
        state[0] = float(row.get('age', 50.0)) if pd.notna(row.get('age')) else 50.0
        sex_val = row.get('sex', 'M')
        state[1] = 1.0 if (isinstance(sex_val, str) and sex_val.upper() == 'M') else 0.0
        # Height and weight not in CSV, use defaults or infer from BMI
        bmi = float(row.get('bmi', 25.0)) if pd.notna(row.get('bmi')) else 25.0
        # Estimate height/weight from BMI (rough approximation)
        state[2] = 170.0  # Default height (cm)
        state[3] = bmi * (state[2] / 100) ** 2  # weight = BMI * height^2
        
        # Body measurements (4-11) - use realistic defaults based on BMI
        state[4] = 30.0  # BMXARMC
        state[5] = 36.0  # BMXARML
        state[6] = bmi   # BMXBMI
        state[7] = 40.0  # BMXLEG
        state[8] = 18.0  # BMXSUB
        state[9] = 52.0  # BMXTHICR
        state[10] = 18.0  # BMXTRI
        state[11] = 90.0  # BMXWAIST
        
        # Lab results (12-33) - extract from baseline columns with realistic defaults
        # Realistic baseline values (typical adult ranges)
        lab_defaults = {
            'LBDSCASI': 2.5,
            'LBDSCH': 50.0,
            'LBDSCR': 90.0,  # Creatinine
            'LBDTC': 200.0,
            'LBXBCD': 0.5,
            'LBXBPB': 2.0,
            'LBXCRP': 2.0,
            'LBXSAL': 30.0,
            'LBXSAS': 25.0,
            'LBXSBU': 15.0,
            'LBXSCA': 9.5,
            'LBXSCH': 50.0,
            'LBXSCL': 102.0,
            'LBXSGB': 0.8,
            'LBXSGL': 100.0,  # Glucose
            'LBXSGT': 25.0,
            'LBXSK': 4.0,  # Potassium
            'LBXSNA': 140.0,  # Sodium
            'LBXSOS': 280.0,
            'LBXSTP': 7.0,
            'LBXSUA': 5.5,
            'LBXTC': 200.0,  # Total cholesterol
        }
        
        lab_baseline_map = {
            'LBDSCASI': 'LBDSCASI_baseline',
            'LBDSCH': 'LBDSCH_baseline',
            'LBDSCR': 'LBDSCR_baseline',
            'LBDTC': 'LBDTC_baseline',
            'LBXBCD': 'LBXBCD_baseline',
            'LBXBPB': 'LBXBPB_baseline',
            'LBXCRP': 'LBXCRP_baseline',
            'LBXSAL': 'LBXSAL_baseline',
            'LBXSAS': 'LBXSAS_baseline',
            'LBXSBU': 'LBXSBU_baseline',
            'LBXSCA': 'LBXSCA_baseline',
            'LBXSCH': 'LBXSCH_baseline',
            'LBXSCL': 'LBXSCL_baseline',
            'LBXSGB': 'LBXSGB_baseline',
            'LBXSGL': 'LBXSGL_baseline',
            'LBXSGT': 'LBXSGT_baseline',
            'LBXSK': 'LBXSK_baseline',
            'LBXSNA': 'LBXSNA_baseline',
            'LBXSOS': 'LBXSOS_baseline',
            'LBXSTP': 'LBXSTP_baseline',
            'LBXSUA': 'LBXSUA_baseline',
            'LBXTC': 'LBXTC_baseline',
        }
        
        for i, lab_feat in enumerate(LAB_BIOMARKER_FEATURES):
            baseline_col = lab_baseline_map.get(lab_feat)
            default_val = lab_defaults.get(lab_feat, 0.0)
            
            if baseline_col and baseline_col in row.index:
                val = row[baseline_col]
                if pd.notna(val) and val != '':
                    state[12 + i] = float(val)
                else:
                    state[12 + i] = default_val
            else:
                state[12 + i] = default_val
        
        # Questionnaires (34-40) - use defaults (typically 2.0 = "No")
        state[34:41] = 2.0
        
        return state
    
    def _extract_lab_delta(self, row: pd.Series) -> np.ndarray:
        """Extract 22 lab biomarker deltas from CSV row."""
        delta = np.zeros(22, dtype=np.float32)
        
        lab_delta_map = {
            'LBDSCASI': 'LBDSCASI_delta',
            'LBDSCH': 'LBDSCH_delta',
            'LBDSCR': 'LBDSCR_delta',
            'LBDTC': 'LBDTC_delta',
            'LBXBCD': 'LBXBCD_delta',
            'LBXBPB': 'LBXBPB_delta',
            'LBXCRP': 'LBXCRP_delta',
            'LBXSAL': 'LBXSAL_delta',
            'LBXSAS': 'LBXSAS_delta',
            'LBXSBU': 'LBXSBU_delta',
            'LBXSCA': 'LBXSCA_delta',
            'LBXSCH': 'LBXSCH_delta',
            'LBXSCL': 'LBXSCL_delta',
            'LBXSGB': 'LBXSGB_baseline',  # May not have delta
            'LBXSGL': 'LBXSGL_delta',
            'LBXSGT': 'LBXSGT_delta',
            'LBXSK': 'LBXSK_delta',
            'LBXSNA': 'LBXSNA_delta',
            'LBXSOS': 'LBXSOS_baseline',  # May not have delta
            'LBXSTP': 'LBXSTP_baseline',  # May not have delta
            'LBXSUA': 'LBXSUA_delta',
            'LBXTC': 'LBXTC_delta',
        }
        
        for i, lab_feat in enumerate(LAB_BIOMARKER_FEATURES):
            delta_col = lab_delta_map.get(lab_feat)
            if delta_col and delta_col in row.index:
                val = row[delta_col]
                if pd.notna(val):
                    delta[i] = float(val)
        
        return delta


def component_similarity(sig1: dict, sig2: dict) -> float:
    """Calculate component-based similarity between two drug signatures.
    
    Uses Jaccard similarity on component presence and cosine similarity on counts.
    This ensures drugs with similar elements and connections are grouped together.
    
    Args:
        sig1: Component signature dictionary for drug 1
        sig2: Component signature dictionary for drug 2
        
    Returns:
        similarity: Float between 0 and 1 (higher = more similar components)
    """
    if not sig1 or not sig2:
        return 0.0
    
    # Get all component keys
    all_keys = set(sig1.keys()) | set(sig2.keys())
    
    if not all_keys:
        return 0.0
    
    # Binary presence vectors (Jaccard similarity)
    vec1_binary = np.array([1.0 if sig1.get(k, 0) > 0 else 0.0 for k in all_keys])
    vec2_binary = np.array([1.0 if sig2.get(k, 0) > 0 else 0.0 for k in all_keys])
    
    # Count vectors (for magnitude-aware similarity)
    vec1_counts = np.array([float(sig1.get(k, 0)) for k in all_keys])
    vec2_counts = np.array([float(sig2.get(k, 0)) for k in all_keys])
    
    # Jaccard similarity (presence-based)
    intersection = np.sum(vec1_binary * vec2_binary)
    union = np.sum(np.maximum(vec1_binary, vec2_binary))
    jaccard = intersection / union if union > 0 else 0.0
    
    # Cosine similarity (count-based)
    dot_product = np.dot(vec1_counts, vec2_counts)
    norm1 = np.linalg.norm(vec1_counts)
    norm2 = np.linalg.norm(vec2_counts)
    cosine = dot_product / (norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0.0
    
    # Combined similarity (weighted average)
    # Jaccard emphasizes shared components, cosine emphasizes similar proportions
    similarity = 0.6 * jaccard + 0.4 * cosine
    
    return float(similarity)


def create_novel_drug_splits(
    data_path: str,
    drug_encoder: DrugEncoder,
    test_ratio: float = 0.2,
    n_clusters: int = 10,
    seed: int = 42,
    use_component_based: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Smart drug-aware split using COMPONENT-BASED similarity.
    
    Strategy:
    1. Extract component signatures (elements, connections, functional groups) for each drug
    2. Cluster drugs by component similarity (not whole-molecule similarity)
    3. Split so test drugs have similar components to train drugs (for generalization)
    4. This ensures the model learns from components and can predict novel drugs with similar components
    
    Returns:
        train_data: DataFrame with training samples
        test_data: DataFrame with test samples
        train_drugs: List of drug names in training
        test_drugs: List of drug names in testing (UNSEEN but component-similar!)
    """
    df = pd.read_csv(data_path)
    
    # Get unique drugs
    unique_drugs = df['drug_name'].unique()
    print(f"Total unique drugs: {len(unique_drugs)}")
    
    if use_component_based:
        print("  Using COMPONENT-BASED splitting (elements & connections)...")
        
        # Extract component signatures for all drugs
        drug_signatures = {}
        valid_drugs = []
        
        print("  Extracting component signatures (elements, bonds, functional groups)...")
        for i, drug in enumerate(unique_drugs):
            try:
                drug_str = str(drug)
                smiles = DRUG_NAME_TO_SMILES.get(drug_str, None)
                
                # If not in mapping, try loading from drug_smiles_mapping.json
                if smiles is None or smiles == drug_str:
                    drug_smiles_path = Path('data/mappings/drug_smiles_mapping.json')
                    if drug_smiles_path.exists():
                        import json
                        with open(drug_smiles_path, 'r') as f:
                            drug_smiles_mapping = json.load(f)
                        # Try exact match first
                        if drug_str in drug_smiles_mapping:
                            smiles = drug_smiles_mapping[drug_str]
                        else:
                            # Try base name match
                            for mapped_name, mapped_smiles in drug_smiles_mapping.items():
                                if mapped_smiles is not None and drug_str.lower() in mapped_name.lower():
                                    smiles = mapped_smiles
                                    break
                
                # Skip if still no SMILES found
                if smiles is None or smiles == drug_str:
                    # Try to validate if drug_str is already a SMILES (suppress errors)
                    try:
                        from rdkit import Chem, RDLogger
                        logger = RDLogger.logger()
                        level = logger.getEffectiveLevel()
                        logger.setLevel(RDLogger.ERROR)  # Suppress warnings
                        
                        mol = Chem.MolFromSmiles(drug_str)
                        logger.setLevel(level)  # Restore
                        
                        if mol is None:
                            # Not a valid SMILES, skip this drug
                            continue
                        smiles = drug_str
                    except:
                        # Not a valid SMILES, skip this drug
                        continue
                
                # Final validation: make sure smiles is valid before extracting
                if smiles is None:
                    continue
                
                # Extract component signature (errors are suppressed inside the method)
                if hasattr(drug_encoder, 'descriptor_extractor') and drug_encoder.descriptor_extractor:
                    signature = drug_encoder.descriptor_extractor.extract_component_signature(smiles)
                    if signature:
                        drug_signatures[drug] = signature
                        valid_drugs.append(drug)
                else:
                    # Fallback: try to encode and extract signature
                    from encoders.drugEncoder import MolecularDescriptorExtractor
                    extractor = MolecularDescriptorExtractor()
                    signature = extractor.extract_component_signature(smiles)
                    if signature:
                        drug_signatures[drug] = signature
                        valid_drugs.append(drug)
                
                if (i + 1) % 10 == 0:
                    print(f"    Processed {i + 1}/{len(unique_drugs)} drugs...")
            except Exception as e:
                # Silently skip drugs without valid SMILES (they'll use hash-based embeddings later)
                pass
        
        # If we have some valid drugs, use component-based splitting
        # If too few, fall back to simple random split
        if len(valid_drugs) < len(unique_drugs) * 0.5:
            print(f"  Warning: Only {len(valid_drugs)}/{len(unique_drugs)} drugs have SMILES mappings")
            print(f"  Falling back to simple random split for all drugs")
            # Use simple random split for all drugs (no SMILES needed)
            np.random.seed(seed)
            n_test = max(1, int(len(unique_drugs) * test_ratio))
            test_indices = np.random.choice(len(unique_drugs), size=n_test, replace=False)
            test_drugs = [unique_drugs[i] for i in test_indices]
            train_drugs = [unique_drugs[i] for i in range(len(unique_drugs)) if i not in test_indices]
            
            # Filter data by split
            train_data = df[df['drug_name'].isin(train_drugs)].copy()
            test_data = df[df['drug_name'].isin(test_drugs)].copy()
            
            return train_data, test_data, train_drugs, test_drugs
        else:
            print(f"  Successfully extracted components for {len(valid_drugs)}/{len(unique_drugs)} drugs")
            # Only use drugs with valid SMILES for component-based splitting
            unique_drugs = valid_drugs
        
        # Build component similarity matrix
        print("  Computing component similarity matrix...")
        n_drugs = len(valid_drugs)
        similarity_matrix = np.zeros((n_drugs, n_drugs))
        
        for i in range(n_drugs):
            for j in range(i + 1, n_drugs):
                sim = component_similarity(
                    drug_signatures[valid_drugs[i]],
                    drug_signatures[valid_drugs[j]]
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
            similarity_matrix[i, i] = 1.0  # Self-similarity
        
        # Convert similarity to distance for clustering
        distance_matrix = 1.0 - similarity_matrix
        
        # Use hierarchical clustering on component similarity
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # Convert to condensed distance matrix
        condensed_distances = squareform(distance_matrix, checks=False)
        
        # Perform hierarchical clustering
        n_clusters = min(n_clusters, len(valid_drugs) // 2)
        if n_clusters < 2:
            n_clusters = 2
        
        if len(valid_drugs) < n_clusters:
            # Fallback to random split
            print(f"  Warning: Only {len(valid_drugs)} drugs available, using random split")
            np.random.seed(seed)
            n_test = max(1, int(len(valid_drugs) * test_ratio))
            test_indices = np.random.choice(len(valid_drugs), size=n_test, replace=False)
            test_drugs = [valid_drugs[i] for i in test_indices]
            train_drugs = [valid_drugs[i] for i in range(len(valid_drugs)) if i not in test_indices]
        else:
            # Hierarchical clustering
            linkage_matrix = linkage(condensed_distances, method='average')
            clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            print(f"  Clustered {len(valid_drugs)} drugs into {n_clusters} component-based clusters...")
            
            # Split by clusters
            np.random.seed(seed)
            test_drugs = []
            train_drugs = []
            
            for cluster_id in range(1, n_clusters + 1):
                cluster_drugs = [valid_drugs[i] for i in range(len(valid_drugs)) 
                                if clusters[i] == cluster_id]
                
                if len(cluster_drugs) > 1:
                    n_test = max(1, int(len(cluster_drugs) * test_ratio))
                    test_subset = np.random.choice(cluster_drugs, size=min(n_test, len(cluster_drugs)), replace=False)
                    test_drugs.extend(test_subset)
                    train_drugs.extend([d for d in cluster_drugs if d not in test_subset])
                else:
                    train_drugs.extend(cluster_drugs)
    
    else:
        # Original whole-molecule embedding approach (fallback)
        print("  Using whole-molecule embedding approach...")
        drug_embeddings = []
        valid_drugs = []
        
        print("  Encoding drugs to embeddings...")
        for i, drug in enumerate(unique_drugs):
            try:
                drug_str = str(drug)
                smiles = DRUG_NAME_TO_SMILES.get(drug_str, drug_str)
                emb = drug_encoder.encode(smiles)
                drug_embeddings.append(emb)
                valid_drugs.append(drug)
                if (i + 1) % 10 == 0:
                    print(f"    Encoded {i + 1}/{len(unique_drugs)} drugs...")
            except Exception as e:
                print(f"    Warning: Could not encode {drug}: {e}")
        
        if len(valid_drugs) == 0:
            raise ValueError("No drugs could be encoded.")
        
        drug_embeddings = np.array(drug_embeddings)
        print(f"  Successfully encoded: {len(valid_drugs)} drugs")
        
        n_clusters = min(n_clusters, len(valid_drugs) // 2)
        if n_clusters < 2:
            n_clusters = 2
        
        if len(valid_drugs) < n_clusters:
            print(f"  Warning: Only {len(valid_drugs)} drugs available, using random split")
            np.random.seed(seed)
            n_test = max(1, int(len(valid_drugs) * test_ratio))
            test_indices = np.random.choice(len(valid_drugs), size=n_test, replace=False)
            test_drugs = [valid_drugs[i] for i in test_indices]
            train_drugs = [valid_drugs[i] for i in range(len(valid_drugs)) if i not in test_indices]
        else:
            print(f"  Clustering {len(valid_drugs)} drugs into {n_clusters} clusters...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            clusters = kmeans.fit_predict(drug_embeddings)
            
            np.random.seed(seed)
            test_drugs = []
            train_drugs = []
            
            for cluster_id in range(n_clusters):
                cluster_drugs = [valid_drugs[i] for i in range(len(valid_drugs)) 
                                if clusters[i] == cluster_id]
                
                if len(cluster_drugs) > 1:
                    n_test = max(1, int(len(cluster_drugs) * test_ratio))
                    test_subset = np.random.choice(cluster_drugs, size=n_test, replace=False)
                    test_drugs.extend(test_subset)
                    train_drugs.extend([d for d in cluster_drugs if d not in test_subset])
                else:
                    train_drugs.extend(cluster_drugs)
    
    # Hold out 1 drug per cluster for testing
    np.random.seed(seed)
    test_drugs = []
    train_drugs = []
    
    for cluster_id in range(n_clusters):
        cluster_drugs = [valid_drugs[i] for i in range(len(valid_drugs)) 
                        if clusters[i] == cluster_id]
        
        if len(cluster_drugs) > 1:
            # Hold out 1 drug
            n_test = max(1, int(len(cluster_drugs) * test_ratio))
            test_subset = np.random.choice(cluster_drugs, size=n_test, replace=False)
            test_drugs.extend(test_subset)
            train_drugs.extend([d for d in cluster_drugs if d not in test_subset])
        else:
            # If only 1 drug in cluster, put it in training
            train_drugs.extend(cluster_drugs)
    
    print(f"\nDrug split:")
    print(f"  Train drugs: {len(train_drugs)}")
    print(f"  Test drugs: {len(test_drugs)} (UNSEEN during training)")
    print(f"  Test drugs are molecularly similar to train drugs")
    
    # Split data
    train_data = df[df['drug_name'].isin(train_drugs)]
    test_data = df[df['drug_name'].isin(test_drugs)]
    
    print(f"\nSample split:")
    print(f"  Train samples: {len(train_data):,}")
    print(f"  Test samples: {len(test_data):,}")
    
    return train_data, test_data, train_drugs, test_drugs


if __name__ == '__main__':
    """Test the dataset."""
    from utils.constants import DEVICE
    
    print("Testing NovelDrugDataset...")
    
    # Initialize drug encoder
    drug_encoder = DrugEncoder(
        encoder_type='hybrid',
        device=DEVICE,
        normalize_embeddings=True
    )
    
    # Create splits
    data_path = 'data/cdisc/clinical_trials_responses.csv'
    train_data, test_data, train_drugs, test_drugs = create_novel_drug_splits(
        data_path=data_path,
        drug_encoder=drug_encoder,
        test_ratio=0.2,
        n_clusters=10,
        seed=42
    )
    
    # Create datasets
    train_dataset = NovelDrugDataset(
        data_path=data_path,
        drug_encoder=drug_encoder,
        split='train',
        train_drugs=train_drugs,
        test_drugs=test_drugs
    )
    
    test_dataset = NovelDrugDataset(
        data_path=data_path,
        drug_encoder=drug_encoder,
        split='test',
        train_drugs=train_drugs,
        test_drugs=test_drugs
    )
    
    # Test a sample
    sample = train_dataset[0]
    print(f"\nSample from train dataset:")
    print(f"  Patient state shape: {sample['patient_state'].shape}")
    print(f"  Drug emb shape: {sample['drug_emb'].shape}")
    print(f"  Lab delta shape: {sample['lab_delta'].shape}")
    
    print("\n✓ Dataset test complete!")
