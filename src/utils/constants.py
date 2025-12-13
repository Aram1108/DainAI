"""
CONFIGURATION FILE FOR GAN-BASED PATIENT GENERATOR

Put all constants and paths here
"""
from pathlib import Path
import torch

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
PREPROC_DIR = BASE_DIR / 'preprocessed_nhanes'
PREPROC_CSV = PREPROC_DIR / 'nhanes_final_complete.csv'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'results'
DRUG_EMBED_DIM = 768          # Dimension of drug embedding vectors
PREDICTOR_HIDDEN = 256        # Hidden layer size for pharmacodynamic predictor

# ============================================================
# GAN HYPERPARAMETERS
# ============================================================
NOISE_DIM = 100           # Latent noise dimension for GAN generator
HIDDEN_DIM = 512          # Hidden layer size for both G and D
BATCH_SIZE = 128          # Training batch size
LEARNING_RATE = 0.0002    # Learning rate (good for GANs)
EPOCHS = 200              # Training epochs

# ============================================================
# DEVICE
# ============================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# FEATURE DEFINITIONS
# ============================================================
USER_INPUT_COLS = ['AGE', 'SEX', 'BMXHT', 'BMXWT']

BODY_MEASUREMENT_COLS = [
    'BMXARMC', 'BMXARML', 'BMXBMI', 'BMXLEG',
    'BMXSUB', 'BMXTHICR', 'BMXTRI', 'BMXWAIST'
]

LAB_RESULT_COLS = [
    'LBDSCASI', 'LBDSCH', 'LBDSCR', 'LBDTC',
    'LBXBCD', 'LBXBPB', 'LBXCRP', 'LBXSAL', 'LBXSAS',
    'LBXSBU', 'LBXSCA', 'LBXSCH', 'LBXSCL', 'LBXSGB',
    'LBXSGL', 'LBXSGT', 'LBXSK', 'LBXSNA', 'LBXSOS',
    'LBXSTP', 'LBXSUA', 'LBXTC'
]

QUESTIONNAIRE_COLS = [
    'MCQ160B', 'MCQ160C', 'MCQ160E', 'MCQ160F',
    'MCQ160K', 'MCQ160L', 'MCQ220'
]

# Total: 8 + 22 + 7 = 37 features to generate
GENERATED_FEATURE_COLS = BODY_MEASUREMENT_COLS + LAB_RESULT_COLS + QUESTIONNAIRE_COLS

# ============================================================
# CREATE DIRECTORIES
# ============================================================
MODEL_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
