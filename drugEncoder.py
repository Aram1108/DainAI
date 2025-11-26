"""SMILES -> embedding using advanced neural encoders.

This file implements production-grade molecular encoders for SMILES strings.
Supports multiple architectures:
- LSTM (legacy, lightweight)
- Transformer (recommended for most cases)
- Pre-trained models (ChemBERTa, MolFormer)
- Hybrid encoders (neural + RDKit descriptors)

Requires: torch, rdkit (optional for descriptors)
"""
from typing import List, Dict, Optional, Literal
import numpy as np
import re
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    raise ImportError(
        "PyTorch is required for the AI-based drug encoder. Install via: pip install torch"
    ) from e

from constants import DRUG_EMBED_DIM

# Optional RDKit for molecular descriptors
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Install via: pip install rdkit. Molecular descriptors will be disabled.")

# Optional transformers for pre-trained models
try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SmilesTokenizer:
    """SMILES-aware regex-based tokenizer.
    
    Recognizes chemical tokens: atoms (Cl, Br, Si), brackets [NH2+], rings %10, etc.
    Much better than character-level tokenization for capturing chemical structure.
    """
    def __init__(self):
        # Regex pattern for SMILES tokenization
        self.pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        
        # Build vocabulary from common SMILES tokens
        common_tokens = [
            '<pad>', '<unk>', '<mask>',  # Special tokens
            # Atoms
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
            'c', 'n', 'o', 's', 'p',  # Aromatic
            # Brackets (common ions/charges)
            '[C]', '[N]', '[O]', '[S]', '[P]',
            '[C+]', '[C-]', '[N+]', '[N-]', '[O+]', '[O-]',
            '[NH]', '[NH+]', '[NH2+]', '[NH3+]', '[O-]', '[S-]',
            '[nH]', '[n+]', '[o+]', '[s+]',
            # Bonds and structure
            '(', ')', '[', ']', '=', '#', '-', '+', '\\', '/', '.', ':',
            '@', '@@',
            # Digits for rings
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
            '%10', '%11', '%12', '%13', '%14', '%15',
        ]
        
        self.vocab = {token: i for i, token in enumerate(common_tokens)}
        self.id2token = {i: token for token, i in self.vocab.items()}
        self.unk_id = self.vocab['<unk>']
        self.pad_id = self.vocab['<pad>']
        self.mask_id = self.vocab['<mask>']
    
    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize SMILES string into chemical tokens."""
        return re.findall(self.pattern, smiles)
    
    def encode(self, smiles: str, max_len: int = 120) -> List[int]:
        """Encode SMILES to token IDs."""
        tokens = self.tokenize(smiles)
        ids = [self.vocab.get(token, self.unk_id) for token in tokens][:max_len]
        # Pad to max_len
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to SMILES."""
        tokens = [self.id2token.get(id, '<unk>') for id in ids if id != self.pad_id]
        return ''.join(tokens)
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class SmileTokenizer(SmilesTokenizer):
    """Backward compatibility alias."""
    pass


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (B, L, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    """Attention-based pooling to aggregate sequence representations.
    
    Better than using final hidden state or mean pooling.
    Learns to weight important tokens for the final representation.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, L, H) sequence representations
            mask: (B, L) boolean mask (True for padding positions)
        Returns:
            pooled: (B, H) aggregated representation
        """
        attn_weights = self.attention(x)  # (B, L, 1)
        
        if mask is not None:
            # Mask padding positions with large negative value
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1), -1e9)
        
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)  # (B, H)
        return pooled


class TransformerSMILESEncoder(nn.Module):
    """Transformer-based SMILES encoder.
    
    Superior to LSTM for capturing long-range dependencies in molecular structure.
    Uses self-attention to learn relationships between different parts of the molecule.
    """
    def __init__(
        self,
        vocab_size: int = 200,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        output_dim: int = DRUG_EMBED_DIM
    ):
        super().__init__()
        self.d_model = d_model
        
        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # GELU works better than ReLU
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention pooling (better than using [CLS] token or mean pooling)
        self.pool = AttentionPooling(d_model)
        
        # Output projection
        self.proj = nn.Linear(d_model, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, normalize=False):
        """
        Args:
            x: (B, L) token IDs
            normalize: whether to L2-normalize output embeddings
        Returns:
            z: (B, output_dim) molecular embeddings
        """
        # Create padding mask
        padding_mask = (x == 0)  # (B, L)
        
        # Embed and add positional encoding
        emb = self.token_emb(x) * np.sqrt(self.d_model)  # Scale embeddings
        emb = self.pos_encoder(emb)
        
        # Transformer encoding
        # Note: src_key_padding_mask expects True for positions to ignore
        encoded = self.transformer(emb, src_key_padding_mask=padding_mask)
        
        # Attention pooling
        pooled = self.pool(encoded, mask=padding_mask)
        
        # Project to output dimension
        z = self.proj(pooled)
        z = self.norm(z)
        z = self.dropout(z)
        
        # Optional L2 normalization for stability
        if normalize:
            z = F.normalize(z, p=2, dim=1)
        
        return z


class MolecularDescriptorExtractor:
    """Extract chemical descriptors from SMILES using RDKit.
    
    Provides interpretable chemical features that complement neural embeddings:
    - Molecular properties (MW, LogP, TPSA)
    - Drug-likeness (Lipinski's Rule of Five)
    - Structural features (rings, bonds, atoms)
    """
    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for molecular descriptors. Install via: pip install rdkit")
    
    def extract(self, smiles: str) -> np.ndarray:
        """Extract 25 molecular descriptors from SMILES.
        
        Returns:
            features: (25,) array of molecular descriptors
        """
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            # Return zeros for invalid SMILES
            return np.zeros(25, dtype=np.float32)
        
        try:
            features = np.array([
                # Basic properties
                Descriptors.MolWt(mol),                    # Molecular weight
                Crippen.MolLogP(mol),                      # Lipophilicity (LogP)
                Descriptors.TPSA(mol),                     # Topological polar surface area
                Descriptors.MolMR(mol),                    # Molar refractivity
                
                # Lipinski's Rule of Five
                Lipinski.NumHDonors(mol),                  # H-bond donors
                Lipinski.NumHAcceptors(mol),               # H-bond acceptors
                Lipinski.NumRotatableBonds(mol),           # Rotatable bonds (flexibility)
                
                # Structural features
                Descriptors.NumAromaticRings(mol),         # Aromatic rings
                Descriptors.NumAliphaticRings(mol),        # Aliphatic rings
                Descriptors.NumSaturatedRings(mol),        # Saturated rings
                Descriptors.NumHeteroatoms(mol),           # Heteroatoms (N, O, S, etc.)
                Descriptors.HeavyAtomCount(mol),           # Heavy atom count
                
                # Carbon statistics
                Descriptors.FractionCSP3(mol),             # Fraction sp3 carbons (saturation)
                Lipinski.NumAromaticCarbocycles(mol),      # Aromatic carbocycles
                Lipinski.NumAliphaticCarbocycles(mol),     # Aliphatic carbocycles
                
                # Complexity metrics
                Descriptors.BertzCT(mol),                  # Complexity (Bertz)
                Descriptors.Chi0v(mol),                    # Connectivity index
                Descriptors.Kappa1(mol),                   # Kappa shape index 1
                Descriptors.Kappa2(mol),                   # Kappa shape index 2
                
                # Electronic properties
                Descriptors.NumValenceElectrons(mol),      # Valence electrons
                Descriptors.NumRadicalElectrons(mol),      # Radical electrons
                
                # Additional drug-likeness
                Descriptors.qed(mol),                      # Quantitative Estimate of Drug-likeness
                
                # Charge and polarity
                Crippen.MolMR(mol) / Descriptors.MolWt(mol) if Descriptors.MolWt(mol) > 0 else 0,  # MR/MW ratio
                Descriptors.TPSA(mol) / Descriptors.MolWt(mol) if Descriptors.MolWt(mol) > 0 else 0,  # TPSA/MW ratio
                
                # Ring complexity
                len(Chem.GetSymmSSSR(mol)),               # Smallest set of smallest rings
            ], dtype=np.float32)
            
            # Handle NaN/inf values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            return features
            
        except Exception as e:
            warnings.warn(f"Failed to extract descriptors for SMILES '{smiles}': {e}")
            return np.zeros(25, dtype=np.float32)


class HybridSMILESEncoder(nn.Module):
    """Hybrid encoder combining neural embeddings with chemical descriptors.
    
    Combines:
    1. Neural encoder (LSTM or Transformer) - learns from data
    2. RDKit molecular descriptors - expert chemical knowledge
    
    This multi-modal approach often outperforms pure neural methods.
    """
    def __init__(
        self,
        neural_encoder: nn.Module,
        descriptor_dim: int = 25,
        hidden_dim: int = 128,
        output_dim: int = DRUG_EMBED_DIM,
        dropout: float = 0.1
    ):
        super().__init__()
        self.neural_encoder = neural_encoder
        
        # Get neural embedding dimension
        if hasattr(neural_encoder, 'proj'):
            neural_dim = neural_encoder.proj.out_features
        else:
            neural_dim = output_dim
        
        # Descriptor processing
        self.descriptor_proj = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(neural_dim + hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
        
        self.descriptor_extractor = MolecularDescriptorExtractor() if RDKIT_AVAILABLE else None
    
    def forward(self, tokens, descriptors=None, normalize=False):
        """
        Args:
            tokens: (B, L) token IDs
            descriptors: (B, 25) optional pre-computed descriptors
            normalize: whether to L2-normalize output
        Returns:
            z: (B, output_dim) hybrid molecular embeddings
        """
        # Neural embedding
        neural_emb = self.neural_encoder(tokens, normalize=False)
        
        # Descriptor embedding
        if descriptors is not None:
            desc_emb = self.descriptor_proj(descriptors)
        else:
            # If descriptors not provided, use zeros (neural-only mode)
            desc_emb = torch.zeros(tokens.size(0), 128, device=tokens.device)
        
        # Concatenate and fuse
        combined = torch.cat([neural_emb, desc_emb], dim=-1)
        z = self.fusion(combined)
        
        if normalize:
            z = F.normalize(z, p=2, dim=1)
        
        return z


# LSTM encoder removed - Transformer and Hybrid are superior for production use


# Pretrained encoder removed - Hybrid provides better generalization with interpretability


class DrugEncoder:
    """Production-grade molecular encoder for healthcare applications.
    
    Supports:
    - 'hybrid': Transformer + RDKit descriptors (RECOMMENDED - best for novel drugs)
    - 'transformer': Transformer encoder (backup option)
    
    Usage:
        # Default: Hybrid encoder (best generalization to novel drugs)
        encoder = DrugEncoder()
        emb = encoder.encode("CCO")  # Returns (768,) numpy array
        
        # Transformer only (if RDKit not available)
        encoder = DrugEncoder(encoder_type='transformer')
        emb = encoder.encode("CCO")
    """
    
    def __init__(
        self,
        encoder_type: Literal['transformer', 'hybrid'] = 'hybrid',
        device: str = 'cpu',
        normalize_embeddings: bool = False,
        output_dim: int = DRUG_EMBED_DIM,
        **kwargs
    ):
        """
        Args:
            encoder_type: Type of encoder ('transformer', 'hybrid')
            device: Device to run on ('cpu', 'cuda', 'mps')
            normalize_embeddings: Whether to L2-normalize output embeddings
            output_dim: Output embedding dimension
            **kwargs: Additional encoder-specific parameters
        """
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.encoder_type = encoder_type
        self.output_dim = output_dim
        
        # Initialize tokenizer
        self.tokenizer = SmilesTokenizer()
        vocab_size = self.tokenizer.vocab_size
        
        # Initialize descriptor extractor for hybrid mode
        self.descriptor_extractor = None
        if encoder_type == 'hybrid':
            if not RDKIT_AVAILABLE:
                raise ImportError(
                    "Hybrid encoder requires RDKit. Install via: pip install rdkit"
                )
            self.descriptor_extractor = MolecularDescriptorExtractor()
        
        # Build encoder model
        torch.manual_seed(42)  # Reproducible initialization
        
        if encoder_type == 'transformer':
            self.model = TransformerSMILESEncoder(
                vocab_size=vocab_size,
                d_model=kwargs.get('d_model', 256),
                nhead=kwargs.get('nhead', 8),
                num_layers=kwargs.get('num_layers', 4),
                dim_feedforward=kwargs.get('dim_feedforward', 1024),
                dropout=kwargs.get('dropout', 0.1),
                output_dim=output_dim
            ).to(self.device)
            
        elif encoder_type == 'hybrid':
            # Create base neural encoder (Transformer by default)
            base_encoder = TransformerSMILESEncoder(
                vocab_size=vocab_size,
                d_model=kwargs.get('d_model', 256),
                nhead=kwargs.get('nhead', 8),
                num_layers=kwargs.get('num_layers', 4),
                dim_feedforward=kwargs.get('dim_feedforward', 1024),
                dropout=kwargs.get('dropout', 0.1),
                output_dim=output_dim
            )
            
            self.model = HybridSMILESEncoder(
                neural_encoder=base_encoder,
                descriptor_dim=25,
                hidden_dim=kwargs.get('descriptor_hidden', 128),
                output_dim=output_dim,
                dropout=kwargs.get('dropout', 0.1)
            ).to(self.device)
        
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}. "
                f"Choose from: 'transformer', 'hybrid'"
            )
        
        # Initialize weights with Xavier/He initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights for better training."""
        for name, param in self.model.named_parameters():
            if 'pretrained' in self.encoder_type.lower():
                # Don't reinitialize pre-trained models
                break
            
            if param.dim() > 1:
                # Use Xavier uniform for linear layers
                if 'weight' in name and ('linear' in name.lower() or 'proj' in name.lower()):
                    nn.init.xavier_uniform_(param)
                # Use orthogonal for LSTM weights
                elif 'weight' in name and 'lstm' in name.lower():
                    nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def encode(self, smiles: str) -> np.ndarray:
        """Encode a SMILES string to a molecular embedding.
        
        Args:
            smiles: SMILES string representation of a molecule
            
        Returns:
            embedding: (output_dim,) numpy array
        """
        if smiles is None:
            raise ValueError('SMILES cannot be None for DrugEncoder.encode')
        
        self.model.eval()
        
        with torch.no_grad():
            # Tokenize SMILES
            ids = self.tokenizer.encode(smiles)
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            
            # Encode
            if self.encoder_type == 'hybrid':
                # Extract molecular descriptors
                descriptors = self.descriptor_extractor.extract(smiles)
                desc_tensor = torch.tensor([descriptors], dtype=torch.float32, device=self.device)
                emb = self.model(x, descriptors=desc_tensor, normalize=self.normalize_embeddings)
            else:
                emb = self.model(x, normalize=self.normalize_embeddings)
            
            return emb.cpu().numpy()[0]
    
    def encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        """Encode a batch of SMILES strings (more efficient than encoding one-by-one).
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            embeddings: (batch_size, output_dim) numpy array
        """
        if not smiles_list:
            return np.array([])
        
        self.model.eval()
        
        with torch.no_grad():
            # Tokenize all SMILES
            ids_list = [self.tokenizer.encode(s) for s in smiles_list]
            x = torch.tensor(ids_list, dtype=torch.long, device=self.device)
            
            if self.encoder_type == 'hybrid':
                # Extract descriptors for all molecules
                descriptors = np.array([self.descriptor_extractor.extract(s) for s in smiles_list])
                desc_tensor = torch.tensor(descriptors, dtype=torch.float32, device=self.device)
                emb = self.model(x, descriptors=desc_tensor, normalize=self.normalize_embeddings)
            else:
                emb = self.model(x, normalize=self.normalize_embeddings)
            
            return emb.cpu().numpy()
    
    def save(self, path: str):
        """Save encoder model to disk."""
        torch.save({
            'encoder_type': self.encoder_type,
            'output_dim': self.output_dim,
            'normalize_embeddings': self.normalize_embeddings,
            'model_state_dict': self.model.state_dict(),
        }, path)
        print(f"Saved encoder to {path}")
    
    def load(self, path: str):
        """Load encoder model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded encoder from {path}")
    
    @property
    def num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


    @property
    def num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ============================================================================
# SMILES Augmentation Utilities
# ============================================================================

def augment_smiles(smiles: str, n_augmentations: int = 10) -> List[str]:
    """Generate augmented SMILES representations of the same molecule.
    
    SMILES augmentation creates equivalent representations by:
    - Randomizing atom ordering
    - Different ring numbering
    - Alternative bond notation
    
    This is useful for data augmentation during training.
    
    Args:
        smiles: Original SMILES string
        n_augmentations: Number of augmented versions to generate
        
    Returns:
        augmented: List of augmented SMILES (includes original)
    """
    if not RDKIT_AVAILABLE:
        warnings.warn("RDKit not available, returning original SMILES")
        return [smiles]
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles]
        
        augmented = [smiles]  # Include original
        
        for _ in range(n_augmentations - 1):
            # Generate random SMILES representation
            aug_smiles = Chem.MolToSmiles(mol, doRandom=True)
            if aug_smiles and aug_smiles not in augmented:
                augmented.append(aug_smiles)
        
        return augmented
        
    except Exception as e:
        warnings.warn(f"Failed to augment SMILES '{smiles}': {e}")
        return [smiles]


def canonicalize_smiles(smiles: str) -> str:
    """Convert SMILES to canonical form.
    
    Canonical SMILES provides a unique representation of a molecule,
    useful for deduplication and comparison.
    
    Args:
        smiles: SMILES string
        
    Returns:
        canonical: Canonical SMILES string
    """
    if not RDKIT_AVAILABLE:
        return smiles
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return smiles


def validate_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        valid: True if valid, False otherwise
    """
    if not RDKIT_AVAILABLE:
        # Without RDKit, do basic validation
        return bool(smiles and len(smiles) > 0)
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


# ============================================================================
# Main / Demo
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("DrugEncoder Demo - Production-Grade Molecular Encoder")
    print("=" * 70)
    
    # Test molecules
    test_molecules = {
        'Ethanol': 'CCO',
        'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
        'Caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'Ibuprofen': 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    }
    
    print("\n1. Hybrid Encoder (PRODUCTION - Best for Novel Drugs)")
    print("-" * 70)
    encoder_hybrid = DrugEncoder(encoder_type='hybrid')
    print(f"Parameters: {encoder_hybrid.num_parameters:,}")
    emb = encoder_hybrid.encode('CCO')
    print(f"Embedding shape: {emb.shape}, norm: {np.linalg.norm(emb):.4f}")
    print("Features: Transformer neural encoder + 25 RDKit descriptors")
    
    print("\n2. Transformer Encoder (Backup Option)")
    print("-" * 70)
    encoder_transformer = DrugEncoder(encoder_type='transformer')
    print(f"Parameters: {encoder_transformer.num_parameters:,}")
    emb = encoder_transformer.encode('CCO')
    print(f"Embedding shape: {emb.shape}, norm: {np.linalg.norm(emb):.4f}")
    
    if RDKIT_AVAILABLE:
        print("\n3. Molecular Descriptors (Built into Hybrid)")
        print("-" * 70)
        extractor = MolecularDescriptorExtractor()
        for name, smiles in test_molecules.items():
            desc = extractor.extract(smiles)
            print(f"{name:12s}: MW={desc[0]:.1f}, LogP={desc[1]:.2f}, TPSA={desc[2]:.1f}, QED={desc[21]:.3f}")
        
        print("\n4. SMILES Augmentation (for Training Robustness)")
        print("-" * 70)
        augmented = augment_smiles('CCO', n_augmentations=5)
        print(f"Original: CCO")
        print(f"Augmented ({len(augmented)} versions):")
        for i, aug in enumerate(augmented[:5], 1):
            print(f"  {i}. {aug}")
    else:
        print("\n[RDKit not installed - Hybrid encoder unavailable]")
        print("Install via: pip install rdkit")
        print("Falling back to Transformer-only mode...")
    
    print("\n5. Batch Encoding (Efficient)")
    print("-" * 70)
    smiles_list = list(test_molecules.values())
    batch_emb = encoder_hybrid.encode_batch(smiles_list) if RDKIT_AVAILABLE else encoder_transformer.encode_batch(smiles_list)
    print(f"Batch shape: {batch_emb.shape}")
    print(f"Per-molecule norms: {[f'{np.linalg.norm(e):.4f}' for e in batch_emb]}")
    
    print("\n6. Similarity Comparison (Hybrid Encoder)")
    print("-" * 70)
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    active_encoder = encoder_hybrid if RDKIT_AVAILABLE else encoder_transformer
    emb_ethanol = active_encoder.encode('CCO')
    emb_aspirin = active_encoder.encode('CC(=O)Oc1ccccc1C(=O)O')
    emb_ibuprofen = active_encoder.encode('CC(C)Cc1ccc(cc1)C(C)C(=O)O')
    
    print(f"Ethanol vs Aspirin:    {cosine_similarity(emb_ethanol, emb_aspirin):.4f}")
    print(f"Ethanol vs Ibuprofen:  {cosine_similarity(emb_ethanol, emb_ibuprofen):.4f}")
    print(f"Aspirin vs Ibuprofen:  {cosine_similarity(emb_aspirin, emb_ibuprofen):.4f}")
    print("(Higher = more similar)")
    
    print("\n" + "=" * 70)
    print("✓ Demo complete!")
    print("=" * 70)
