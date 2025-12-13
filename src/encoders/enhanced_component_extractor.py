"""Enhanced component extraction with 200-dimensional molecular fingerprints.

This module extracts comprehensive component signatures from SMILES strings:

- Functional groups (30): hydroxyl, carboxyl, amines, etc.
- Ring systems (30): benzene, pyridine, heterocycles, etc.
- Atom types (20): element counts and connectivity
- Connectivity patterns (40): bond types between elements
- Topology (30): 3D structure hints
- Physicochemical (30): drug-like properties
- Pharmacophore (20): binding motifs

Total: 200 features that represent ANY drug molecule
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import warnings

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Install via: pip install rdkit")

class EnhancedComponentExtractor:
    """Extract comprehensive 200-dim component signatures from SMILES.
    
    This is the CORE of the component-based approach. Every drug is
    represented as a combination of these 200 fundamental components.
    """
    
    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required. Install via: pip install rdkit")
        
        # Define functional group SMARTS patterns
        self.functional_groups = self._init_functional_groups()
        
        # Define ring system patterns
        self.ring_systems = self._init_ring_systems()
        
        # Component name mapping (for interpretability)
        self.component_names = self._init_component_names()
    
    def _init_functional_groups(self) -> Dict[str, str]:
        """Define 30 functional group SMARTS patterns."""
        return {
            # Oxygen-containing
            'hydroxyl': '[OH]',
            'phenol': 'c[OH]',
            'alcohol': '[C][OH]',
            'carboxyl': 'C(=O)[OH]',
            'ester': 'C(=O)[O][C]',
            'ether': '[C][O][C]',
            'carbonyl': '[C]=[O]',
            'aldehyde': '[CH]=O',
            'ketone': '[C](=O)[C]',
            
            # Nitrogen-containing
            'amine_primary': '[NH2]',
            'amine_secondary': '[NH][C]',
            'amine_tertiary': '[N]([C])([C])[C]',
            'aniline': 'c[NH2]',
            'amide': 'C(=O)[NH]',
            'nitrile': 'C#N',
            'nitro': '[N+](=O)[O-]',
            'imine': 'C=[N]',
            
            # Sulfur-containing
            'thiol': '[SH]',
            'sulfide': '[S][C]',
            'disulfide': '[S][S]',
            'sulfonyl': 'S(=O)(=O)',
            'sulfate': 'S(=O)(=O)([O])[O]',
            
            # Phosphorus-containing
            'phosphate': 'P(=O)([O])([O])[O]',
            'phosphonate': 'P(=O)([O])[O]',
            
            # Halogen-containing
            'alkyl_fluoride': '[C][F]',
            'alkyl_chloride': '[C][Cl]',
            'aryl_halide': 'c[F,Cl,Br,I]',
            
            # Other
            'epoxide': 'C1OC1',
            'lactone': 'C(=O)OC',
            'lactam': 'C(=O)NC',
        }
    
    def _init_ring_systems(self) -> Dict[str, str]:
        """Define 30 ring system SMARTS patterns."""
        return {
            # Aromatic 6-member
            'benzene': 'c1ccccc1',
            'pyridine': 'n1ccccc1',
            'pyrimidine': 'n1cnccc1',
            'pyrazine': 'n1ccncc1',
            'triazine': 'n1ncncn1',
            
            # Aromatic 5-member
            'pyrrole': 'n1cccc1',
            'furan': 'o1cccc1',
            'thiophene': 's1cccc1',
            'imidazole': 'n1cncc1',
            'pyrazole': 'n1ncc1',
            'thiazole': 's1cncc1',
            'oxazole': 'o1cncc1',
            
            # Saturated rings
            'cyclohexane': 'C1CCCCC1',
            'cyclopentane': 'C1CCCC1',
            'piperidine': 'N1CCCCC1',
            'piperazine': 'N1CCNCC1',
            'morpholine': 'O1CCNCC1',
            'pyrrolidine': 'N1CCCC1',
            
            # Fused systems
            'indole': 'c1ccc2[nH]ccc2c1',
            'quinoline': 'c1ccc2ncccc2c1',
            'isoquinoline': 'c1ccc2cnccc2c1',
            'naphthalene': 'c1ccc2ccccc2c1',
            'purine': 'n1cnc2ncnc2c1',
            
            # Other
            'aziridine': 'C1CN1',
            'oxirane': 'C1CO1',
            'thiirane': 'C1CS1',
            'azetidine': 'C1CNC1',
            'oxetane': 'C1COC1',
            'tetrahydrofuran': 'C1COCC1',
            'dioxane': 'C1COCCO1',
        }
    
    def _init_component_names(self) -> List[str]:
        """Create ordered list of all 200 component names."""
        names = []
        
        # Functional groups (30)
        names.extend([f'fg_{name}' for name in self.functional_groups.keys()])
        
        # Ring systems (30)
        names.extend([f'ring_{name}' for name in self.ring_systems.keys()])
        
        # Atom types (20)
        names.extend(['atom_C', 'atom_N', 'atom_O', 'atom_S', 'atom_P',
                     'atom_F', 'atom_Cl', 'atom_Br', 'atom_I', 'atom_H',
                     'atom_C_sp3_frac', 'atom_C_aromatic_frac',
                     'atom_heavy', 'atom_hetero', 'atom_chiral',
                     'atom_aromatic', 'atom_aliphatic',
                     'atom_charged_pos', 'atom_charged_neg', 'atom_radical'])
        
        # Connectivity (40)
        names.extend(['bond_C_C_single', 'bond_C_C_double', 'bond_C_C_triple', 'bond_C_C_aromatic',
                     'bond_C_N_single', 'bond_C_N_double', 'bond_C_N_triple', 'bond_C_N_aromatic',
                     'bond_C_O_single', 'bond_C_O_double', 'bond_C_O_aromatic',
                     'bond_C_S_single', 'bond_C_S_double', 'bond_N_N', 'bond_N_O',
                     'bond_O_O', 'bond_S_S', 'bond_C_F', 'bond_C_Cl', 'bond_C_Br', 'bond_C_I',
                     'bond_C_P', 'bond_P_O', 'bond_S_O', 'bond_S_N',
                     'bond_avg_order', 'bond_total', 'bond_rotatable', 'bond_conjugated',
                     'bond_aromatic_total', 'bond_aliphatic_total', 'bond_ring', 'bond_exocyclic',
                     'bond_endocyclic', 'bond_bridge', 'bond_fusion',
                     'conn_avg_degree', 'conn_max_degree', 'conn_min_degree', 'conn_branching'])
        
        # Topology (30)
        names.extend(['topo_complexity', 'topo_wiener', 'topo_balaban',
                     'topo_kappa1', 'topo_kappa2', 'topo_kappa3',
                     'topo_chi0', 'topo_chi1', 'topo_chi2', 'topo_chi3',
                     'topo_chi0v', 'topo_chi1v', 'topo_flexibility',
                     'topo_rings_total', 'topo_rings_aromatic', 'topo_rings_aliphatic',
                     'topo_rings_saturated', 'topo_rings_fused', 'topo_rings_spiro',
                     'topo_rings_bridgehead', 'topo_rings_max_size', 'topo_rings_min_size',
                     'topo_rings_avg_size', 'topo_globularity', 'topo_asphericity',
                     'topo_eccentricity', 'topo_radius_gyration',
                     'topo_symmetry', 'topo_stereo_centers', 'topo_undefined_stereo'])
        
        # Physicochemical (30)
        names.extend(['phys_MW', 'phys_logP', 'phys_logD', 'phys_pKa_acid', 'phys_pKa_base',
                     'phys_TPSA', 'phys_PSA', 'phys_MR', 'phys_polarizability',
                     'phys_HBD', 'phys_HBA', 'phys_fraction_csp3', 'phys_fraction_rotatable',
                     'phys_aromatic_prop', 'phys_aliphatic_prop',
                     'phys_QED', 'phys_SA_score', 'phys_druglikeness',
                     'phys_lipinski_violations', 'phys_lipinski_HBA', 'phys_lipinski_HBD',
                     'phys_lipinski_MW', 'phys_lipinski_logP', 'phys_lipinski_rotatable',
                     'phys_veber_rotatable', 'phys_veber_TPSA',
                     'phys_ghose_MW', 'phys_ghose_logP', 'phys_ghose_atoms', 'phys_ghose_MR'])
        
        # Pharmacophore (20)
        names.extend(['pharm_hydrophobic', 'pharm_aromatic', 'pharm_Hdonor', 'pharm_Hacceptor',
                     'pharm_positive', 'pharm_negative', 'pharm_metal_binding',
                     'pharm_pi_stacking', 'pharm_cation_pi', 'pharm_anion_pi',
                     'pharm_halogen_donor', 'pharm_halogen_acceptor',
                     'pharm_CH_pi', 'pharm_OH_pi', 'pharm_NH_pi', 'pharm_SH_pi',
                     'pharm_electrophilic', 'pharm_nucleophilic',
                     'pharm_radical', 'pharm_reactive'])
        
        assert len(names) == 200, f"Expected 200 components, got {len(names)}"
        
        return names
    
    def extract_component_vector(self, smiles: str) -> np.ndarray:
        """Extract complete 200-dim component vector from SMILES.
        
        This is the MAIN method that converts any drug SMILES → 200 features.
        
        Args:
            smiles: SMILES string
        
        Returns:
            components: (200,) numpy array
        """
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            warnings.warn(f"Invalid SMILES: {smiles}")
            return np.zeros(200, dtype=np.float32)
        
        components = []
        
        # 1. Functional groups (30 features)
        components.extend(self._extract_functional_groups(mol))
        
        # 2. Ring systems (30 features)
        components.extend(self._extract_ring_systems(mol))
        
        # 3. Atom types (20 features)
        components.extend(self._extract_atom_types(mol))
        
        # 4. Connectivity (40 features)
        components.extend(self._extract_connectivity(mol))
        
        # 5. Topology (30 features)
        components.extend(self._extract_topology(mol))
        
        # 6. Physicochemical (30 features)
        components.extend(self._extract_physicochemical(mol))
        
        # 7. Pharmacophore (20 features)
        components.extend(self._extract_pharmacophore(mol))
        
        components = np.array(components, dtype=np.float32)
        
        # Handle NaN/inf
        components = np.nan_to_num(components, nan=0.0, posinf=0.0, neginf=0.0)
        
        assert components.shape[0] == 200, f"Expected 200 components, got {components.shape[0]}"
        
        return components
    
    def _extract_functional_groups(self, mol) -> List[float]:
        """Extract 30 functional group counts."""
        counts = []
        for name, pattern in self.functional_groups.items():
            try:
                patt_mol = Chem.MolFromSmarts(pattern)
                if patt_mol:
                    matches = mol.GetSubstructMatches(patt_mol)
                    counts.append(float(len(matches)))
                else:
                    counts.append(0.0)
            except:
                counts.append(0.0)
        
        return counts
    
    def _extract_ring_systems(self, mol) -> List[float]:
        """Extract 30 ring system counts."""
        counts = []
        for name, pattern in self.ring_systems.items():
            try:
                patt_mol = Chem.MolFromSmarts(pattern)
                if patt_mol:
                    matches = mol.GetSubstructMatches(patt_mol)
                    counts.append(float(len(matches)))
                else:
                    counts.append(0.0)
            except:
                counts.append(0.0)
        
        return counts
    
    def _extract_atom_types(self, mol) -> List[float]:
        """Extract 20 atom type features."""
        atom_counts = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
        
        features = []
        
        # Element counts (10)
        for element in ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']:
            features.append(float(atom_counts.get(element, 0)))
        
        # Carbon statistics (2)
        features.append(Descriptors.FractionCSP3(mol))
        features.append(float(Lipinski.NumAromaticCarbocycles(mol)) / max(1, atom_counts.get('C', 1)))
        
        # Other counts (8)
        features.append(float(mol.GetNumHeavyAtoms()))
        features.append(float(Descriptors.NumHeteroatoms(mol)))
        features.append(float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))))
        features.append(float(sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())))
        features.append(float(sum(1 for atom in mol.GetAtoms() if not atom.GetIsAromatic())))
        features.append(float(sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() > 0)))
        features.append(float(sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() < 0)))
        features.append(float(Descriptors.NumRadicalElectrons(mol)))
        
        return features
    
    def _extract_connectivity(self, mol) -> List[float]:
        """Extract 40 connectivity features."""
        features = []
        
        # Bond type counts (25)
        bond_counts = {}
        for bond in mol.GetBonds():
            begin_symbol = bond.GetBeginAtom().GetSymbol()
            end_symbol = bond.GetEndAtom().GetSymbol()
            bond_type = str(bond.GetBondType())
            
            # Create sorted pair key
            pair = tuple(sorted([begin_symbol, end_symbol]))
            key = (pair[0], pair[1], bond_type)
            bond_counts[key] = bond_counts.get(key, 0) + 1
        
        # Specific bond types (25 features)
        for key in [('C', 'C', 'SINGLE'), ('C', 'C', 'DOUBLE'), ('C', 'C', 'TRIPLE'), ('C', 'C', 'AROMATIC'),
                    ('C', 'N', 'SINGLE'), ('C', 'N', 'DOUBLE'), ('C', 'N', 'TRIPLE'), ('C', 'N', 'AROMATIC'),
                    ('C', 'O', 'SINGLE'), ('C', 'O', 'DOUBLE'), ('C', 'O', 'AROMATIC'),
                    ('C', 'S', 'SINGLE'), ('C', 'S', 'DOUBLE'),
                    ('N', 'N', 'SINGLE'), ('N', 'O', 'SINGLE'),
                    ('O', 'O', 'SINGLE'), ('S', 'S', 'SINGLE'),
                    ('C', 'F', 'SINGLE'), ('C', 'Cl', 'SINGLE'), ('C', 'Br', 'SINGLE'), ('C', 'I', 'SINGLE'),
                    ('C', 'P', 'SINGLE'), ('O', 'P', 'SINGLE'), ('O', 'S', 'SINGLE'), ('N', 'S', 'SINGLE')]:
            features.append(float(bond_counts.get(key, 0)))
        
        # Bond statistics (15 features)
        all_bonds = list(mol.GetBonds())
        if all_bonds:
            bond_orders = [bond.GetBondTypeAsDouble() for bond in all_bonds]
            features.append(np.mean(bond_orders))  # Average bond order
        else:
            features.append(0.0)
        
        features.append(float(mol.GetNumBonds()))
        features.append(float(Lipinski.NumRotatableBonds(mol)))
        features.append(float(sum(1 for bond in all_bonds if bond.GetIsConjugated())))
        features.append(float(sum(1 for bond in all_bonds if bond.GetIsAromatic())))
        features.append(float(sum(1 for bond in all_bonds if not bond.GetIsAromatic())))
        
        ring_info = mol.GetRingInfo()
        ring_bonds = set()
        for ring in ring_info.BondRings():
            ring_bonds.update(ring)
        
        features.append(float(len(ring_bonds)))
        features.append(float(mol.GetNumBonds() - len(ring_bonds)))  # Exocyclic
        features.append(float(len(ring_bonds)))  # Endocyclic (same as ring bonds)
        features.append(float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)))
        features.append(float(ring_info.NumRings()))
        
        # Connectivity statistics (4 features)
        degrees = [atom.GetDegree() for atom in mol.GetAtoms()]
        if degrees:
            features.append(np.mean(degrees))
            features.append(float(max(degrees)))
            features.append(float(min(degrees)))
            features.append(np.std(degrees))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return features
    
    def _extract_topology(self, mol) -> List[float]:
        """Extract 30 topology features."""
        features = []
        
        # Complexity indices (10)
        features.append(Descriptors.BertzCT(mol))
        try:
            features.append(Chem.GraphDescriptors.BalabanJ(mol))
        except:
            features.append(0.0)
        
        features.append(Chem.GraphDescriptors.Kappa1(mol))
        features.append(Chem.GraphDescriptors.Kappa2(mol))
        features.append(Chem.GraphDescriptors.Kappa3(mol))
        
        features.append(Chem.GraphDescriptors.Chi0(mol))
        features.append(Chem.GraphDescriptors.Chi1(mol))
        features.append(Chem.GraphDescriptors.Chi2n(mol))
        features.append(Chem.GraphDescriptors.Chi3n(mol))
        features.append(Chem.GraphDescriptors.Chi0v(mol))
        # Removed Chi1v to keep exactly 10 complexity indices (was 11, causing 201 total features)
        
        # Flexibility (1)
        features.append(float(Lipinski.NumRotatableBonds(mol)) / max(1, mol.GetNumBonds()))
        
        # Ring statistics (10)
        ring_info = mol.GetRingInfo()
        features.append(float(ring_info.NumRings()))
        features.append(float(rdMolDescriptors.CalcNumAromaticRings(mol)))
        features.append(float(rdMolDescriptors.CalcNumAliphaticRings(mol)))
        features.append(float(rdMolDescriptors.CalcNumSaturatedRings(mol)))
        
        ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
        if ring_sizes:
            features.append(float(sum(1 for r in ring_info.AtomRings() if len(set(r)) < len(r))))  # Fused
            features.append(float(rdMolDescriptors.CalcNumSpiroAtoms(mol)))
            features.append(float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)))
            features.append(float(max(ring_sizes)))
            features.append(float(min(ring_sizes)))
            features.append(np.mean(ring_sizes))
        else:
            features.extend([0.0] * 6)
        
        # 3D-ish features (9)
        try:
            # Note: These require 3D coordinates, using 2D approximations
            features.append(1.0)  # Globularity placeholder
            features.append(0.5)  # Asphericity placeholder
            features.append(1.0)  # Eccentricity placeholder
            features.append(float(mol.GetNumAtoms() ** 0.5))  # Radius of gyration approx
            
            # Symmetry and stereo
            features.append(float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=False))))
            features.append(float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))))
            features.append(float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)) - 
                                  len(Chem.FindMolChiralCenters(mol, includeUnassigned=False))))
            
            # Symmetry classes
            try:
                symmetry_classes = len(set(Chem.CanonicalRankAtoms(mol)))
                features.append(float(symmetry_classes) / max(1, mol.GetNumAtoms()))
            except:
                features.append(0.5)
            
            # Planarity approximation
            features.append(float(rdMolDescriptors.CalcNumAromaticRings(mol)) / max(1, ring_info.NumRings()))
        except:
            features.extend([0.0] * 9)
        
        return features
    
    def _extract_physicochemical(self, mol) -> List[float]:
        """Extract 30 physicochemical features."""
        features = []
        
        # Basic properties (9)
        features.append(Descriptors.MolWt(mol))
        features.append(Crippen.MolLogP(mol))
        features.append(Crippen.MolLogP(mol))  # logD approximation (same as logP for now)
        
        # pKa (not readily available in RDKit, use placeholders)
        features.append(7.0)  # pKa acid placeholder
        features.append(7.0)  # pKa base placeholder
        
        features.append(Descriptors.TPSA(mol))
        features.append(Descriptors.TPSA(mol))  # PSA (same as TPSA)
        features.append(Crippen.MolMR(mol))
        
        # Polarizability approximation
        features.append(Crippen.MolMR(mol) / max(1, Descriptors.MolWt(mol)))
        
        # H-bond (2)
        features.append(float(Lipinski.NumHDonors(mol)))
        features.append(float(Lipinski.NumHAcceptors(mol)))
        
        # Fractions (2)
        features.append(Descriptors.FractionCSP3(mol))
        features.append(float(Lipinski.NumRotatableBonds(mol)) / max(1, mol.GetNumBonds()))
        
        # Aromatic/aliphatic proportions (2)
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        features.append(aromatic_atoms / max(1, mol.GetNumAtoms()))
        features.append(1.0 - aromatic_atoms / max(1, mol.GetNumAtoms()))
        
        # Drug-likeness (3)
        features.append(Descriptors.qed(mol))
        
        # SA score (synthetic accessibility)
        try:
            from rdkit.Chem import rdMolDescriptors
            # SA score not directly available, use complexity as proxy
            features.append(min(10.0, Descriptors.BertzCT(mol) / 100.0))
        except:
            features.append(5.0)
        
        # Drug-likeness score (placeholder)
        features.append(Descriptors.qed(mol))
        
        # Lipinski violations (1)
        violations = 0
        if Descriptors.MolWt(mol) > 500:
            violations += 1
        if Crippen.MolLogP(mol) > 5:
            violations += 1
        if Lipinski.NumHDonors(mol) > 5:
            violations += 1
        if Lipinski.NumHAcceptors(mol) > 10:
            violations += 1
        features.append(float(violations))
        
        # Lipinski components (4)
        features.append(float(Lipinski.NumHAcceptors(mol)))
        features.append(float(Lipinski.NumHDonors(mol)))
        features.append(Descriptors.MolWt(mol))
        features.append(Crippen.MolLogP(mol))
        features.append(float(Lipinski.NumRotatableBonds(mol)))
        
        # Veber (2)
        features.append(float(Lipinski.NumRotatableBonds(mol)))
        features.append(Descriptors.TPSA(mol))
        
        # Ghose (3)
        features.append(Descriptors.MolWt(mol))
        features.append(Crippen.MolLogP(mol))
        features.append(float(mol.GetNumAtoms()))
        features.append(Crippen.MolMR(mol))
        
        return features
    
    def _extract_pharmacophore(self, mol) -> List[float]:
        """Extract 20 pharmacophore features."""
        features = []
        
        # Hydrophobic centers (count of lipophilic atoms)
        hydrophobic = sum(1 for atom in mol.GetAtoms() 
                         if atom.GetSymbol() in ['C'] and not atom.GetIsAromatic())
        features.append(float(hydrophobic))
        
        # Aromatic centers
        aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        features.append(float(aromatic))
        
        # H-bond donors and acceptors
        features.append(float(Lipinski.NumHDonors(mol)))
        features.append(float(Lipinski.NumHAcceptors(mol)))
        
        # Ionizable groups (approximation)
        # Positive ionizable (amines)
        pos_ion = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NH2,NH1,NH0+]')))
        features.append(float(pos_ion))
        
        # Negative ionizable (carboxylates, sulfonates, phosphates)
        neg_ion = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[O-,S-,P-]')))
        features.append(float(neg_ion))
        
        # Metal binding (chelating groups: carboxylate, phosphate, thiol)
        metal_binding = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[C](=O)[O-]')))
        metal_binding += len(mol.GetSubstructMatches(Chem.MolFromSmarts('[SH]')))
        features.append(float(metal_binding))
        
        # Pi-stacking capable (aromatic rings)
        features.append(float(rdMolDescriptors.CalcNumAromaticRings(mol)))
        
        # Cation-pi (aromatic + positive charge nearby)
        cation_pi = min(aromatic, pos_ion)
        features.append(float(cation_pi))
        
        # Anion-pi (approximation)
        anion_pi = min(aromatic, neg_ion)
        features.append(float(anion_pi))
        
        # Halogen bonding
        halogen_donor = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in ['Cl', 'Br', 'I'])
        features.append(float(halogen_donor))
        
        halogen_acceptor = Lipinski.NumHAcceptors(mol)  # Approximation
        features.append(float(halogen_acceptor))
        
        # CH-pi, OH-pi, NH-pi, SH-pi (approximations)
        features.append(float(hydrophobic * aromatic / max(1, mol.GetNumAtoms())))
        features.append(float(len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH]'))) * aromatic / max(1, mol.GetNumAtoms())))
        features.append(float(len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NH]'))) * aromatic / max(1, mol.GetNumAtoms())))
        features.append(float(len(mol.GetSubstructMatches(Chem.MolFromSmarts('[SH]'))) * aromatic / max(1, mol.GetNumAtoms())))
        
        # Electrophilic/nucleophilic centers (approximations)
        electrophilic = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[C](=O)')))  # Carbonyls
        features.append(float(electrophilic))
        
        nucleophilic = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NH2,OH,SH]')))
        features.append(float(nucleophilic))
        
        # Radical sites and reactive sites (approximations)
        features.append(float(Descriptors.NumRadicalElectrons(mol)))
        
        reactive = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[C]=[C]')))  # Double bonds
        reactive += len(mol.GetSubstructMatches(Chem.MolFromSmarts('[C]#[C]')))  # Triple bonds
        features.append(float(reactive))
        
        return features
    
    def summarize_components(self, smiles: str) -> str:
        """Generate human-readable summary of component profile.
        
        Useful for interpretability and debugging.
        
        Returns:
            summary: Multi-line string describing key components
        """
        components = self.extract_component_vector(smiles)
        
        lines = []
        lines.append(f"Component Summary for: {smiles}")
        lines.append("=" * 60)
        
        # Find top functional groups
        fg_start, fg_end = 0, 30
        fg_values = components[fg_start:fg_end]
        top_fg_indices = np.argsort(fg_values)[::-1][:5]
        
        lines.append("\nTop Functional Groups:")
        for idx in top_fg_indices:
            if fg_values[idx] > 0:
                name = self.component_names[idx].replace('fg_', '')
                lines.append(f"  • {name}: {int(fg_values[idx])}")
        
        # Find top ring systems
        ring_start, ring_end = 30, 60
        ring_values = components[ring_start:ring_end]
        top_ring_indices = np.argsort(ring_values)[::-1][:5]
        
        lines.append("\nTop Ring Systems:")
        for idx in top_ring_indices:
            if ring_values[idx] > 0:
                name = self.component_names[ring_start + idx].replace('ring_', '')
                lines.append(f"  • {name}: {int(ring_values[idx])}")
        
        # Key physicochemical properties
        phys_start = 140
        mw = components[phys_start]
        logp = components[phys_start + 1]
        tpsa = components[phys_start + 5]
        qed = components[phys_start + 15]
        
        lines.append("\nPhysicochemical Properties:")
        lines.append(f"  • Molecular Weight: {mw:.1f} Da")
        lines.append(f"  • LogP: {logp:.2f}")
        lines.append(f"  • TPSA: {tpsa:.1f} ų")
        lines.append(f"  • QED (Drug-likeness): {qed:.3f}")
        
        # Atom counts
        atom_start = 60
        c_count = components[atom_start]
        n_count = components[atom_start + 1]
        o_count = components[atom_start + 2]
        
        lines.append("\nAtom Composition:")
        lines.append(f"  • C: {int(c_count)}, N: {int(n_count)}, O: {int(o_count)}")
        
        return "\n".join(lines)
    
    def validate_component_vector(self, components: np.ndarray) -> bool:
        """Check if component vector represents chemically valid molecule.
        
        Basic sanity checks:
        - All values are non-negative
        - Atom counts are reasonable
        - Bond counts consistent with atom counts
        
        Returns:
            valid: True if passes basic checks
        """
        if components.shape[0] != 200:
            return False
        
        if np.any(components < 0):
            return False
        
        if np.any(np.isnan(components)) or np.any(np.isinf(components)):
            return False
        
        # Check atom counts are reasonable (< 1000 atoms total)
        atom_start = 60
        total_atoms = np.sum(components[atom_start:atom_start+10])
        if total_atoms > 1000:
            return False
        
        return True


# Convenience function
def extract_components_from_smiles(smiles: str) -> np.ndarray:
    """Convenience function to extract components from SMILES string.
    
    Args:
        smiles: SMILES string
    
    Returns:
        components: (200,) numpy array
    """
    extractor = EnhancedComponentExtractor()
    return extractor.extract_component_vector(smiles)


if __name__ == '__main__':
    """Test component extraction."""
    print("Testing Enhanced Component Extractor\n")
    
    # Test molecules
    test_smiles = {
        'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
        'Metformin': 'CN(C)C(=N)NC(=N)N',
        'Ibuprofen': 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
        'Caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    }
    
    extractor = EnhancedComponentExtractor()
    for name, smiles in test_smiles.items():
        print(f"\n{name}:")
        print("-" * 60)
        components = extractor.extract_component_vector(smiles)
        print(f"Component vector shape: {components.shape}")
        print(f"Non-zero components: {np.sum(components > 0)}")
        print(f"Valid: {extractor.validate_component_vector(components)}")
        print(f"\n{extractor.summarize_components(smiles)}")

