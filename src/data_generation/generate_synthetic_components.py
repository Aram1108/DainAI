"""Generate synthetic component library with diverse and toxic profiles.

Generates 100,000 component profiles representing:
- 60,000 from diverse drug classes (antibiotics, antiinflammatories, etc.)
- 30,000 random valid combinations
- 10,000 TOXIC patterns (hepatotoxic, nephrotoxic, cardiotoxic)

CRITICAL: Toxic examples teach model to recognize dangerous structures!
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from encoders.enhanced_component_extractor import EnhancedComponentExtractor

class SyntheticComponentGenerator:
    """Generate chemically valid synthetic component profiles."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.component_extractor = EnhancedComponentExtractor()
        self.component_names = self.component_extractor.component_names
        
        # Chemical validity constraints
        self.constraints = {
            'min_MW': 100,
            'max_MW': 800,
            'min_C': 1,
            'max_C': 50,
            'max_rings': 8,
            'max_halogens': 15,
            'max_N': 20,
            'max_O': 20,
            'max_S': 10,
        }
    
    def generate_random_valid(self, n_samples: int = 1000) -> np.ndarray:
        """Generate random but chemically valid component vectors.
        
        Algorithm:
        1. Sample MW from drug-like distribution (200-600 Da)
        2. Allocate atoms based on MW
        3. Add functional groups probabilistically
        4. Add ring systems
        5. Validate chemical feasibility
        """
        components_list = []
        
        for _ in tqdm(range(n_samples), desc="Generating random valid"):
            # Start with zeros
            components = np.zeros(200, dtype=np.float32)
            
            # 1. Sample MW and basic properties
            mw = np.random.normal(400, 150)
            mw = np.clip(mw, self.constraints['min_MW'], self.constraints['max_MW'])
            components[140] = mw  # phys_MW
            
            # Estimate atom counts from MW (rough heuristic)
            total_heavy = int(mw / 12)  # Assume average atom weight ~12
            c_count = np.random.binomial(total_heavy, 0.7)  # 70% carbon
            n_count = np.random.binomial(total_heavy - c_count, 0.3)
            o_count = np.random.binomial(total_heavy - c_count - n_count, 0.5)
            s_count = np.random.poisson(0.5)
            
            # Atom types (indices 60-69)
            components[60] = min(c_count, self.constraints['max_C'])
            components[61] = min(n_count, self.constraints['max_N'])
            components[62] = min(o_count, self.constraints['max_O'])
            components[63] = min(s_count, self.constraints['max_S'])
            
            # 2. Add functional groups (indices 0-29)
            # Hydroxyl
            if np.random.rand() < 0.3:
                components[0] = np.random.poisson(2)
            
            # Carboxyl
            if np.random.rand() < 0.2:
                components[3] = np.random.poisson(1)
            
            # Amines
            if n_count > 0:
                amine_prob = min(0.4, n_count / 10)
                if np.random.rand() < amine_prob:
                    components[9] = np.random.poisson(1)  # primary
                    components[10] = np.random.poisson(1)  # secondary
            
            # 3. Add ring systems (indices 30-59)
            n_rings = np.random.poisson(2)
            n_rings = min(n_rings, self.constraints['max_rings'])
            
            if n_rings > 0:
                # Benzene rings common
                if np.random.rand() < 0.6:
                    components[30] = np.random.poisson(1.5)
                
                # Pyridine
                if np.random.rand() < 0.2:
                    components[31] = np.random.poisson(0.5)
            
            # 4. Add physicochemical properties
            # LogP
            components[141] = np.random.normal(2.0, 1.5)
            components[141] = np.clip(components[141], -2, 6)
            
            # TPSA
            tpsa = o_count * 20 + n_count * 15  # Rough estimate
            components[145] = tpsa
            
            # HBD and HBA
            components[149] = min(components[0] + components[9], 10)  # HBD
            components[150] = min(o_count + n_count, 15)  # HBA
            
            # QED (drug-likeness)
            components[155] = np.random.beta(5, 2)  # Biased toward drug-like
            
            # 5. Validate and adjust
            if self._validate_basic(components):
                components_list.append(components)
        
        return np.array(components_list, dtype=np.float32)
    
    def generate_drug_class(self, drug_class: str, n_samples: int = 10000) -> np.ndarray:
        """Generate components biased toward specific drug class.
        
        Drug classes:
        - antibiotics: β-lactams, quinolones, macrolides
        - antiinflammatories: NSAIDs, COX inhibitors
        - antidiabetics: biguanides, sulfonylureas, PPAR agonists
        - antihypertensives: ACE inhibitors, beta-blockers, diuretics
        - lipid_lowering: statins, fibrates
        - cns_drugs: crosses BBB (lipophilic, small MW)
        """
        base_components = self.generate_random_valid(n_samples)
        
        if drug_class == 'antibiotics':
            # β-lactam-like: amides, lactam rings
            base_components[:, 13] += np.random.poisson(1, n_samples)  # amide
            base_components[:, 29] += np.random.poisson(0.5, n_samples)  # lactam
            
        elif drug_class == 'antiinflammatories':
            # NSAID-like: carboxyl + aromatic rings
            base_components[:, 3] += np.random.poisson(1, n_samples)  # carboxyl
            base_components[:, 30] += np.random.poisson(1.5, n_samples)  # benzene
            base_components[:, 141] += 1.0  # More lipophilic
            
        elif drug_class == 'antidiabetics':
            # Biguanide-like or sulfonylurea-like
            if np.random.rand() < 0.5:
                # Biguanide (metformin-like): lots of nitrogen
                base_components[:, 61] += np.random.poisson(5, n_samples)  # N count
                base_components[:, 9] += np.random.poisson(2, n_samples)  # amines
            else:
                # Sulfonylurea: sulfonyl + urea
                base_components[:, 20] += 1  # sulfonyl
                base_components[:, 13] += 1  # amide/urea-like
            
        elif drug_class == 'antihypertensives':
            # ACE inhibitor-like: carboxyl + aromatic
            base_components[:, 3] += np.random.poisson(1, n_samples)
            base_components[:, 30] += np.random.poisson(1, n_samples)
            
        elif drug_class == 'lipid_lowering':
            # Statin-like: complex structure, multiple rings
            base_components[:, 30] += np.random.poisson(2, n_samples)  # rings
            base_components[:, 0] += np.random.poisson(3, n_samples)  # hydroxyls
            
        elif drug_class == 'cns_drugs':
            # BBB-crossing: lipophilic, smaller MW
            base_components[:, 140] *= 0.8  # Lower MW
            base_components[:, 141] += 0.5  # More lipophilic
            base_components[:, 61] += np.random.poisson(2, n_samples)  # N-rich
        
        return base_components
    
    def generate_toxic_patterns(self, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate TOXIC component patterns with toxicity labels.
        
        CRITICAL: Model must learn to recognize dangerous structures!
        
        Toxicity types:
        - Hepatotoxic (liver damage): halogenated aromatics, reactive metabolites
        - Nephrotoxic (kidney damage): heavy metals, aminoglycosides
        - Cardiotoxic (heart damage): prolonged QT, arrhythmias
        - Neurotoxic: crosses BBB with reactive groups
        - Carcinogenic: PAHs, nitrosamines, aflatoxins
        
        Returns:
            components: (n_samples, 200)
            toxicity_labels: (n_samples, 5) - binary labels for each toxicity type
        """
        print("\n⚠️  GENERATING TOXIC COMPONENT PATTERNS")
        print("These teach the model to recognize dangerous structures\n")
        
        components_list = []
        toxicity_labels = []
        
        for _ in tqdm(range(n_samples), desc="Generating toxic patterns"):
            # Start with base valid structure
            components = self.generate_random_valid(1)[0]
            toxicity = np.zeros(5, dtype=np.float32)  # [hepato, nephro, cardio, neuro, carcino]
            
            # Randomly choose toxicity type(s)
            tox_type = np.random.choice(['hepato', 'nephro', 'cardio', 'neuro', 'carcino'])
            
            if tox_type == 'hepato':
                # Hepatotoxic: multiple halogens + aromatic rings
                components[66] += np.random.poisson(3)  # Cl
                components[67] += np.random.poisson(2)  # Br
                components[30] += np.random.poisson(2)  # benzene
                components[141] += 2.0  # Very lipophilic
                components[140] += 100  # Larger MW
                toxicity[0] = 1.0
                
            elif tox_type == 'nephro':
                # Nephrotoxic: heavy metals (simulated), aminoglycosides
                components[61] += np.random.poisson(8)  # Lots of nitrogen
                components[9] += np.random.poisson(4)  # Many amines
                components[0] += np.random.poisson(6)  # Many hydroxyls
                components[140] += 200  # Very large
                toxicity[1] = 1.0
                
            elif tox_type == 'cardio':
                # Cardiotoxic: aromatic amines, prolonged structures
                components[61] += np.random.poisson(4)  # N-rich
                components[30] += np.random.poisson(3)  # Multiple aromatics
                components[11] = np.random.poisson(2)  # aniline
                components[141] += 1.5  # Lipophilic
                toxicity[2] = 1.0
                
            elif tox_type == 'neuro':
                # Neurotoxic: BBB-crossing + reactive groups
                components[140] = np.random.uniform(150, 300)  # Small MW
                components[141] = np.random.uniform(3, 5)  # Very lipophilic
                components[15] = np.random.poisson(1)  # nitro groups
                components[27] = np.random.poisson(1)  # epoxides (reactive)
                toxicity[3] = 1.0
                
            elif tox_type == 'carcino':
                # Carcinogenic: PAHs (fused rings), nitrosamines
                components[30] += np.random.poisson(4)  # Many benzenes
                components[51] = np.random.poisson(2)  # naphthalene (fused)
                components[15] = np.random.poisson(1)  # nitro
                components[114] += 2.0  # Increased ring fusion
                toxicity[4] = 1.0
            
            components_list.append(components)
            toxicity_labels.append(toxicity)
        
        return np.array(components_list, dtype=np.float32), np.array(toxicity_labels, dtype=np.float32)
    
    def _validate_basic(self, components: np.ndarray) -> bool:
        """Basic validation of component vector."""
        # No negative values
        if np.any(components < 0):
            return False
        
        # MW in range
        mw = components[140]
        if mw < self.constraints['min_MW'] or mw > self.constraints['max_MW']:
            return False
        
        # At least some carbon
        if components[60] < self.constraints['min_C']:
            return False
        
        return True
    
    def generate_diverse_library(
        self,
        n_total: int = 100000,
        include_toxic: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate complete diverse component library.
        
        Breakdown:
        - 10K antibiotics
        - 10K antiinflammatories
        - 10K antidiabetics
        - 10K antihypertensives
        - 10K lipid lowering
        - 10K CNS drugs
        - 30K random valid
        - 10K toxic (if include_toxic=True)
        
        Returns:
            components: (n_total, 200)
            toxicity_labels: (n_total, 5) or None
        """
        print("\n" + "="*70)
        print("GENERATING DIVERSE COMPONENT LIBRARY")
        print("="*70)
        
        components_list = []
        toxicity_list = []
        
        # Drug classes (60K)
        drug_classes = [
            'antibiotics', 'antiinflammatories', 'antidiabetics',
            'antihypertensives', 'lipid_lowering', 'cns_drugs'
        ]
        
        for drug_class in drug_classes:
            print(f"\nGenerating {drug_class}...")
            comps = self.generate_drug_class(drug_class, n_samples=10000)
            components_list.append(comps)
            
            # Non-toxic labels
            toxicity_list.append(np.zeros((10000, 5), dtype=np.float32))
        
        # Random valid (30K)
        print("\nGenerating random valid...")
        comps = self.generate_random_valid(n_samples=30000)
        components_list.append(comps)
        toxicity_list.append(np.zeros((30000, 5), dtype=np.float32))
        
        # Toxic (10K)
        if include_toxic:
            print("\nGenerating toxic patterns...")
            toxic_comps, toxic_labels = self.generate_toxic_patterns(n_samples=10000)
            components_list.append(toxic_comps)
            toxicity_list.append(toxic_labels)
        
        # Stack all
        all_components = np.vstack(components_list)
        all_toxicity = np.vstack(toxicity_list) if include_toxic else None
        
        # Shuffle
        indices = np.random.permutation(len(all_components))
        all_components = all_components[indices]
        if all_toxicity is not None:
            all_toxicity = all_toxicity[indices]
        
        # Print statistics
        print("\n" + "="*70)
        print("LIBRARY STATISTICS")
        print("="*70)
        print(f"\nTotal components: {len(all_components):,}")
        print(f"Dimension: 200")
        print(f"Toxicity labels included: {include_toxic}")
        
        if include_toxic:
            print(f"\nToxicity breakdown:")
            print(f"  Hepatotoxic: {np.sum(all_toxicity[:, 0]):.0f}")
            print(f"  Nephrotoxic: {np.sum(all_toxicity[:, 1]):.0f}")
            print(f"  Cardiotoxic: {np.sum(all_toxicity[:, 2]):.0f}")
            print(f"  Neurotoxic: {np.sum(all_toxicity[:, 3]):.0f}")
            print(f"  Carcinogenic: {np.sum(all_toxicity[:, 4]):.0f}")
        
        # Component statistics
        print(f"\nComponent Statistics:")
        print(f"  MW: {all_components[:, 140].mean():.1f} ± {all_components[:, 140].std():.1f} Da")
        print(f"  LogP: {all_components[:, 141].mean():.2f} ± {all_components[:, 141].std():.2f}")
        print(f"  TPSA: {all_components[:, 145].mean():.1f} ± {all_components[:, 145].std():.1f} ų")
        print(f"  QED: {all_components[:, 155].mean():.3f} ± {all_components[:, 155].std():.3f}")
        
        # Aromatic rings
        print(f"  Benzene rings: {all_components[:, 30].mean():.2f} ± {all_components[:, 30].std():.2f}")
        print(f"  Functional groups/molecule: {all_components[:, :30].sum(axis=1).mean():.2f}")
        
        return all_components, all_toxicity

def main():
    """Main execution."""
    # Create output directory
    output_dir = Path('data/synthetic')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate library
    generator = SyntheticComponentGenerator(seed=42)
    components, toxicity_labels = generator.generate_diverse_library(
        n_total=100000,
        include_toxic=True
    )
    
    # Save
    output_path = output_dir / 'component_library.npz'
    np.savez_compressed(
        output_path,
        components=components,
        toxicity_labels=toxicity_labels,
        component_names=generator.component_names,
        metadata={
            'n_samples': len(components),
            'n_features': 200,
            'includes_toxic': True,
            'generation_method': 'diverse_drug_class_with_toxic',
        }
    )
    
    print(f"\n✓ Saved component library to: {output_path}")
    print(f"  Size: {components.nbytes / 1e6:.1f} MB")
    print("\nNext step: Run 04_simulate_effects.py")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()

