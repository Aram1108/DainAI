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

# Human-readable display names for lab metrics (for saved results and graphs)
LAB_CODE_TO_DISPLAY_NAME = {
    'LBDSCASI': 'Sedation Level Score',
    'LBDSCH': 'Cholinesterase (enzyme that breaks down nerve signals)',
    'LBDSCR': 'Creatinine (waste product filtered by kidneys)',
    'LBDTC': 'Total Cholesterol (derived)',
    'LBXBCD': 'Blood Cadmium (heavy metal toxin level in blood)',
    'LBXBPB': 'Blood Lead (lead poisoning marker)',
    'LBXCRP': 'C-Reactive Protein (inflammation marker)',
    'LBXSAL': 'Albumin (main protein in blood)',
    'LBXSAS': 'AST (liver enzyme)',
    'LBXSBU': 'Blood Urea Nitrogen (kidney waste marker)',
    'LBXSCA': 'Calcium (mineral level in blood)',
    'LBXSCH': 'Total Cholesterol',
    'LBXSCL': 'Chloride (salt balance in blood)',
    'LBXSGB': 'Globulin (immune system proteins in blood)',
    'LBXSGL': 'Blood Sugar (glucose)',
    'LBXSGT': 'GGT (liver/bile enzyme)',
    'LBXSK': 'Potassium (electrolyte)',
    'LBXSNA': 'Sodium (electrolyte)',
    'LBXSOS': 'Blood Osmolality (how concentrated the blood is)',
    'LBXSTP': 'Total Protein in blood',
    'LBXSUA': 'Uric Acid (waste product from cell breakdown)',
    'LBXTC': 'Total Cholesterol (alternate measurement)',
}


def get_metric_display_name(code: str) -> str:
    """Return human-readable name for a lab code, or the code itself if unknown."""
    base = code.replace('_delta', '').strip()
    return LAB_CODE_TO_DISPLAY_NAME.get(base, code)


QUESTIONNAIRE_COLS = [
    'MCQ160B', 'MCQ160C', 'MCQ160E', 'MCQ160F',
    'MCQ160K', 'MCQ160L', 'MCQ220'
]

# Total: 8 + 22 + 7 = 37 features to generate
GENERATED_FEATURE_COLS = BODY_MEASUREMENT_COLS + LAB_RESULT_COLS + QUESTIONNAIRE_COLS

# ============================================================
# LAB REFERENCE RANGES
# ============================================================
# Clinical lab test reference ranges are defined in:
# src/utils/lab_reference_ranges.py
# Use: from utils.lab_reference_ranges import LAB_RANGES, get_lab_range, validate_lab_value

# ============================================================
# CREATE DIRECTORIES
# ============================================================
MODEL_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
