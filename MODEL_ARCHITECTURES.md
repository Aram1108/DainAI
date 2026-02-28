# Model Architectures Documentation

This document provides a comprehensive overview of all neural network architectures used in the BloodTwin project.

## Notation

Throughout this document, tensor dimensions are denoted using the following notation:
- **B**: Batch size (number of samples processed together in a single forward pass)
- **L**: Sequence length (for sequential data like SMILES strings)
- **H**: Hidden dimension
- **D**: Feature dimension

For example:
- `(B, 768)` means a tensor with shape `[batch_size, 768]` - one 768-dimensional vector per sample in the batch
- `(B, 41, 256)` means a tensor with shape `[batch_size, 41, 256]` - 41 features of 256 dimensions each, for each sample in the batch

---

## 1. Patient Generator GAN (Conditional GAN)

**File:** `src/models/patient_generator_gan.py`

**Purpose:** Generates synthetic patient profiles based on user inputs (age, sex, height, weight)

### Architecture Overview

#### Generator Network
- **Input:** 
  - Conditions: 4 features (age, sex, height, weight)
  - Noise: 100 dimensions
  - **Total input:** 104 dimensions
- **Output:** 37 features (8 body measurements + 22 lab results + 7 questionnaire responses)
- **Architecture:**
  ```
  104 → 512 → 512 → 256 → 37
  ```
- **Layers:**
  1. Linear(104, 512) + BatchNorm1d + LeakyReLU(0.2) + Dropout(0.3)
  2. Linear(512, 512) + BatchNorm1d + LeakyReLU(0.2) + Dropout(0.3)
  3. Linear(512, 256) + BatchNorm1d + LeakyReLU(0.2) + Dropout(0.3)
  4. Linear(256, 37) + Tanh()

#### Discriminator Network
- **Input:** 41 features (4 conditions + 37 generated features)
- **Output:** 1 probability (real/fake)
- **Architecture:**
  ```
  41 → 256 → 128 → 1
  ```
- **Layers:**
  1. Linear(41, 256) + LeakyReLU(0.2) + Dropout(0.5)
  2. Linear(256, 128) + LeakyReLU(0.2) + Dropout(0.5)
  3. Linear(128, 1) + Sigmoid()

### Training Details
- **Loss:** Binary Cross Entropy (BCE)
- **Optimizer:** Adam (betas=0.5, 0.999)
- **Learning Rate:** Generator: 0.0003, Discriminator: 0.0002
- **Batch Size:** 128
- **Evaluation Metric:** FID (Fréchet Inception Distance) - lower is better

### Generated Features
- **Body Measurements (8):** BMXARMC, BMXARML, BMXBMI, BMXLEG, BMXSUB, BMXTHICR, BMXTRI, BMXWAIST
- **Lab Results (22):** LBDSCASI, LBDSCH, LBDSCR, LBDTC, LBXBCD, LBXBPB, LBXCRP, LBXSAL, LBXSAS, LBXSBU, LBXSCA, LBXSCH, LBXSCL, LBXSGB, LBXSGL, LBXSGT, LBXSK, LBXSNA, LBXSOS, LBXSTP, LBXSUA, LBXTC
- **Questionnaires (7):** MCQ160B, MCQ160C, MCQ160E, MCQ160F, MCQ160K, MCQ160L, MCQ220

---

## 2. Pharmacodynamic Predictor (Cross-Attention)

**File:** `src/models/pharmacodynamicPredictor.py`

**Purpose:** Predicts changes to 22 lab biomarkers given patient state and drug embedding

### Architecture: CrossAttentionPredictor

#### Input
- **Patient State:** 41 features (4 demographics + 8 body + 22 labs + 7 questionnaires)
- **Drug Embedding:** 768 dimensions (from DrugEncoder)

#### Output
- **Lab Biomarker Changes:** 22 features (delta values for lab biomarkers only)

#### Architecture Flow

1. **Drug Projection**
   - Input: (B, 768)
   - Output: (B, 1, 256)
   - Layers: Linear(768, 256) + LayerNorm + GELU + Dropout(0.1)

2. **Patient Feature Projection**
   - Input: (B, 41)
   - Output: (B, 41, 256)
   - Each patient feature is projected to hidden dimension: Linear(1, 256) + LayerNorm + GELU + Dropout(0.1)

3. **Multi-Layer Cross-Attention** (4 layers)
   - **Cross-Attention:** Drug queries patient features
     - Query: Drug embedding (B, 1, 256)
     - Key/Value: Patient features (B, 41, 256)
     - Multi-head attention: 8 heads
     - Output: (B, 1, 256)
   - **Residual Connection:** attended + attn_out
   - **Layer Normalization**
   - **Feedforward Network:**
     - Linear(256, 1024) + GELU + Dropout(0.1)
     - Linear(1024, 256) + Dropout(0.05)
   - **Residual Connection:** attended + ff_out
   - **Layer Normalization**

4. **Prediction Head**
   - Input: (B, 256)
   - Output: (B, 22)
   - Layers:
     - Linear(256, 128) + LayerNorm + GELU + Dropout(0.1)
     - Linear(128, 64) + LayerNorm + GELU + Dropout(0.05)
     - Linear(64, 22)

### Configuration
- **Hidden Dimension:** 256
- **Number of Heads:** 8
- **Number of Layers:** 4
- **Dropout:** 0.1
- **Total Parameters:** ~2.6M

### Features
- **Physiological Constraints:** Enforces bounds on lab values
- **Uncertainty Quantification:** Monte Carlo Dropout for confidence intervals
- **Attention Visualization:** Returns attention weights for interpretability

---

## 3. Time Series Predictor (Transformer)

**File:** `src/models/time_series_predictor.py`

**Purpose:** Converts static drug response predictions into temporal trajectories

**Note:** This model ONLY converts static predictions (from PharmacodynamicPredictor) into time series. It does NOT predict drug effects.

### Architecture: TimeSeriesTransformer

#### Input
- **Patient Features:** 5 features (age, sex, bmi, adherence, days_on_drug)
- **Drug ID:** Integer (mapped from drug vocabulary)
- **Baselines:** (B, num_metrics) - baseline lab values
- **Final Deltas:** (B, num_metrics) - final delta from PharmacodynamicPredictor

#### Output
- **Predicted Deltas:** (B, num_timepoints, num_metrics)
- **Uncertainties:** (B, num_timepoints, num_metrics) - optional

#### Architecture Flow

1. **Input Encoding**
   - Drug Embedding: Embedding(drug_vocab_size, 64)
   - Input Concatenation: patient_features + drug_embedding + baselines + final_deltas
   - Input Dimension: patient_feature_dim + 64 + num_metrics * 2
   - Projection: Linear(input_dim, 256)

2. **Timepoint Embeddings**
   - Learnable embeddings for each timepoint (e.g., 10, 20, 30, 60, 90, 180 days)
   - Embedding(num_timepoints, 256)

3. **Positional Encoding**
   - Sinusoidal positional encoding for temporal relationships

4. **Transformer Encoder** (4 layers)
   - **Encoder Layer:**
     - Self-attention: 8 heads, hidden_dim=256
     - Feedforward: 256 → 1024 → 256 (expansion factor 4)
     - Dropout: 0.1
   - **Number of Layers:** 4

5. **Output Projection**
   - Linear(256, num_metrics) - predicts delta at each timepoint

6. **Uncertainty Head** (optional)
   - Linear(256, 128) + ReLU + Dropout(0.1)
   - Linear(128, num_metrics) - predicts log(std)

### Configuration
- **Hidden Dimension:** 256
- **Number of Heads:** 8
- **Number of Layers:** 4
- **Dropout:** 0.1
- **Timepoints:** [10, 20, 30, 60, 90, 180] days (default)

### Features
- **Drug-Specific Trajectories:** Learns different response profiles for different drugs
- **Adherence Modeling:** Accounts for patient adherence in trajectory shape
- **Multi-Metric Prediction:** Predicts all lab values simultaneously
- **Uncertainty Quantification:** Provides confidence intervals

---

## 4. Drug Encoder (Hybrid Neural + RDKit)

**File:** `src/encoders/drugEncoder.py`

**Purpose:** Encodes SMILES strings into 768-dimensional embeddings

### Architecture Options

#### Option 1: Transformer Encoder (Recommended)
**Class:** `TransformerSMILESEncoder`

- **Input:** SMILES string (tokenized)
- **Output:** 768-dimensional embedding
- **Architecture:**
  1. **Token Embedding:** Embedding(vocab_size, 256)
  2. **Positional Encoding:** Sinusoidal positional encoding
  3. **Transformer Encoder:**
     - Layers: 4
     - Heads: 8
     - Hidden: 256
     - Feedforward: 1024
     - Activation: GELU
  4. **Attention Pooling:** Aggregates sequence representation
  5. **Output Projection:** Linear(256, 768) + LayerNorm

#### Option 2: LSTM Encoder (Legacy)
- **Architecture:** LSTM(256) → Linear(768)
- **Note:** Less effective than Transformer for long-range dependencies

#### Option 3: Hybrid Encoder (Recommended for Production)
**Class:** `DrugEncoder` with `encoder_type='hybrid'`

- **Neural Component:** TransformerSMILESEncoder (512 dims)
- **RDKit Descriptors:** 256 molecular descriptors
- **Concatenation:** 512 + 256 = 768 dimensions

**RDKit Descriptors Include:**
- Molecular properties (MW, LogP, TPSA, etc.)
- Drug-likeness (Lipinski's Rule of Five)
- Structural features (rings, bonds, atoms)
- Component features (functional groups, substructures)

### Tokenization
- **Tokenizer:** `SmilesTokenizer`
- **Method:** Regex-based SMILES-aware tokenization
- **Vocabulary:** ~200 tokens (atoms, bonds, brackets, rings, etc.)

### Configuration
- **Output Dimension:** 768 (DRUG_EMBED_DIM)
- **Max Sequence Length:** 120 tokens
- **Vocabulary Size:** ~200 tokens

---

## Model Pipeline Flow

```
1. Patient Generator GAN
   Input: age, sex, height, weight
   Output: Complete patient profile (41 features)

2. Drug Encoder
   Input: SMILES string
   Output: 768-dimensional drug embedding

3. Pharmacodynamic Predictor
   Input: Patient state (41) + Drug embedding (768)
   Output: Static lab biomarker changes (22 deltas)

4. Time Series Predictor
   Input: Static predictions + patient features + drug info
   Output: Temporal trajectories (deltas at 10, 20, 30, 60, 90, 180 days)
```

---

## Summary Statistics

| Model | Input Dim | Output Dim | Parameters | Architecture Type |
|-------|-----------|------------|------------|-------------------|
| Generator | 104 | 37 | ~500K | Feedforward (4 layers) |
| Discriminator | 41 | 1 | ~100K | Feedforward (3 layers) |
| Pharmacodynamic Predictor | 41 + 768 | 22 | ~2.6M | Cross-Attention Transformer |
| Time Series Predictor | Variable | Variable | ~1-2M | Transformer Encoder |
| Drug Encoder (Transformer) | Variable | 768 | ~500K-1M | Transformer Encoder |
| Drug Encoder (Hybrid) | Variable | 768 | ~500K-1M | Transformer + RDKit |

---

## Key Design Principles

1. **Modularity:** Each model has a specific, well-defined purpose
2. **Interpretability:** Attention mechanisms allow visualization of model decisions
3. **Physiological Constraints:** Models enforce realistic bounds on predictions
4. **Uncertainty Quantification:** Models provide confidence intervals where applicable
5. **Scalability:** Transformer architectures handle variable-length inputs efficiently

---

## Training Requirements

- **Patient Generator GAN:** Requires NHANES dataset (~23K patients)
- **Pharmacodynamic Predictor:** Requires synthetic drug-response pairs
- **Time Series Predictor:** Requires temporal trajectory data
- **Drug Encoder:** Pre-trained or trained on molecular datasets

---

*Last Updated: Based on current codebase structure*

