"""Fetch SMILES strings for CDISC drugs from public databases.

This script:
1. Reads drug names from CDISC data (adsl.csv)
2. Queries PubChem, DrugBank, and ChEMBL APIs
3. Saves drug→SMILES mapping to data/mappings/drug_smiles_mapping.json
4. Validates SMILES with RDKit

Usage:
    python scripts/01_fetch_drug_smiles.py
"""
import sys
import json
import time
from pathlib import Path
from typing import Dict, Optional
import warnings
import pandas as pd
import requests
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
# Also add project root for data access
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. SMILES validation disabled.")

class DrugSMILESFetcher:
    """Fetch SMILES strings from multiple public databases."""
    
    def __init__(self):
        self.cache = {}
        self.manual_overrides = {
            # Add manual SMILES for drugs not in databases
            'Placebo': None,
            'Screen Failure': None,
        }
    
    def fetch_from_pubchem(self, drug_name: str) -> Optional[str]:
        """Fetch SMILES from PubChem REST API."""
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/CanonicalSMILES/JSON"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                return smiles
        except Exception as e:
            pass
        
        return None
    
    def fetch_from_chembl(self, drug_name: str) -> Optional[str]:
        """Fetch SMILES from ChEMBL API."""
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={drug_name}&format=json"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('molecules') and len(data['molecules']) > 0:
                    return data['molecules'][0].get('molecule_structures', {}).get('canonical_smiles')
        except Exception as e:
            pass
        
        return None
    
    def clean_drug_name(self, drug_name: str) -> str:
        """Clean drug name for API queries.
        
        Removes dose information, formulation details, etc.
        Examples:
            'Xanomeline Low Dose' → 'Xanomeline'
            'Placebo' → 'Placebo'
        """
        # Remove common suffixes
        for suffix in [' Low Dose', ' High Dose', ' Extended Release', ' XR', 
                       ' Tablet', ' Capsule', ' Injection']:
            drug_name = drug_name.replace(suffix, '')
        
        return drug_name.strip()
    
    def validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES string with RDKit."""
        if not RDKIT_AVAILABLE:
            return True  # Assume valid if RDKit not available
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def fetch_smiles(self, drug_name: str, verbose: bool = True) -> Optional[str]:
        """Fetch SMILES from all available sources.
        
        Priority: Manual overrides → PubChem → ChEMBL
        """
        # Check manual overrides first
        if drug_name in self.manual_overrides:
            return self.manual_overrides[drug_name]
        
        # Check cache
        if drug_name in self.cache:
            return self.cache[drug_name]
        
        # Clean name
        clean_name = self.clean_drug_name(drug_name)
        
        if verbose:
            print(f"  Fetching: {drug_name} (cleaned: {clean_name})")
        
        # Try PubChem first
        smiles = self.fetch_from_pubchem(clean_name)
        if smiles and self.validate_smiles(smiles):
            if verbose:
                print(f"    ✓ Found in PubChem: {smiles[:50]}...")
            self.cache[drug_name] = smiles
            return smiles
        
        # Try ChEMBL
        time.sleep(0.5)  # Rate limiting
        smiles = self.fetch_from_chembl(clean_name)
        if smiles and self.validate_smiles(smiles):
            if verbose:
                print(f"    ✓ Found in ChEMBL: {smiles[:50]}...")
            self.cache[drug_name] = smiles
            return smiles
        
        if verbose:
            print(f"    ✗ Not found")
        
        return None
    
    def fetch_for_cdisc_data(self, cdisc_path: str = 'data/cdisc/adsl.csv') -> Dict[str, Optional[str]]:
        """Extract unique drug names from CDISC data and fetch SMILES.
        
        Returns:
            mapping: {drug_name: smiles_string or None}
        """
        print("\n" + "="*70)
        print("FETCHING DRUG SMILES FROM PUBLIC DATABASES")
        print("="*70)
        
        # Read CDISC data
        adsl = pd.read_csv(cdisc_path)
        
        # Extract unique treatment names
        # Try different column names (CDISC naming varies)
        treatment_col = None
        for col in ['TRT01P', 'ARM', 'TRTP', 'ACTARM']:
            if col in adsl.columns:
                treatment_col = col
                break
        
        if treatment_col is None:
            raise ValueError("Could not find treatment column in CDISC data")
        
        unique_drugs = adsl[treatment_col].unique()
        print(f"\nFound {len(unique_drugs)} unique treatments in CDISC data:")
        for drug in unique_drugs:
            print(f"  • {drug}")
        
        # Fetch SMILES for each drug
        print(f"\nFetching SMILES from PubChem and ChEMBL...\n")
        
        mapping = {}
        for drug in tqdm(unique_drugs, desc="Fetching SMILES"):
            smiles = self.fetch_smiles(drug, verbose=False)
            mapping[drug] = smiles
            time.sleep(0.5)  # Rate limiting
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        found = sum(1 for v in mapping.values() if v is not None)
        print(f"\nSuccessfully found SMILES: {found}/{len(mapping)}")
        
        print("\nResults:")
        print(f"{'Drug Name':<30} {'SMILES Found':<15} {'SMILES Length'}")
        print("-"*70)
        for drug, smiles in mapping.items():
            if smiles:
                print(f"{drug:<30} ✓{'':<14} {len(smiles)}")
            else:
                print(f"{drug:<30} ✗{'':<14} N/A")
        
        # Warn about missing SMILES
        missing = [drug for drug, smiles in mapping.items() if smiles is None]
        if missing:
            print("\n⚠️  WARNING: Missing SMILES for:")
            for drug in missing:
                print(f"    • {drug}")
            print("\nYou may need to:")
            print("  1. Add manual SMILES to manual_overrides dict")
            print("  2. Search drug database websites manually")
            print("  3. Use alternative drug names")
        
        return mapping

def main():
    """Main execution."""
    # Create output directory
    output_dir = Path('data/mappings')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch SMILES
    fetcher = DrugSMILESFetcher()
    mapping = fetcher.fetch_for_cdisc_data()
    
    # Save to JSON
    output_path = output_dir / 'drug_smiles_mapping.json'
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\n✓ Saved drug→SMILES mapping to: {output_path}")
    print("\nNext step: Run 02_preprocess_cdisc_data.py")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()

