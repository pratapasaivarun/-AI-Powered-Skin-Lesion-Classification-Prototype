"""
Configuration settings for the Skin Cancer Detection System.
All hyperparameters, paths, and class metadata are centralized here.
"""

import os

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
METADATA_CSV = os.path.join(DATASET_DIR, "HAM10000_metadata.csv")
IMAGE_DIR_PART1 = os.path.join(DATASET_DIR, "HAM10000_images_part_1")
IMAGE_DIR_PART2 = os.path.join(DATASET_DIR, "HAM10000_images_part_2")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Saved model paths
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "skin_cancer_best.keras")
FINETUNED_MODEL_PATH = os.path.join(MODELS_DIR, "skin_cancer_finetuned_best.keras")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "skin_cancer_model.keras")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
TRAINING_HISTORY_PATH = os.path.join(RESULTS_DIR, "training_history.pkl")

# ============================================================
# IMAGE & TRAINING HYPERPARAMETERS
# ============================================================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 40          # Frozen backbone training
EPOCHS_STAGE2 = 10          # Fine-tuning
LEARNING_RATE_STAGE1 = 1e-4
LEARNING_RATE_STAGE2 = 1e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 5
LR_REDUCE_PATIENCE = 3
LR_REDUCE_FACTOR = 0.2
RANDOM_STATE = 42
TEST_SIZE = 0.3
VAL_TEST_SPLIT = 0.5
NUM_CLASSES = 7

# Data Augmentation
ROTATION_RANGE = 20
ZOOM_RANGE = 0.2
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
HORIZONTAL_FLIP = True
FILL_MODE = "nearest"

# ============================================================
# MODEL ARCHITECTURES
# ============================================================
MODEL_TYPES = ["custom_cnn", "mobilenet_v2", "hybrid"]

# MobileNetV2 fine-tuning: unfreeze top N layers
MOBILENET_UNFREEZE_LAYERS = 50
HYBRID_UNFREEZE_LAYERS = 30

# ============================================================
# CLASS METADATA (HAM10000 — 7 Lesion Types)
# ============================================================
CLASS_INFO = {
    "akiec": {
        "risk": "RED",
        "name": "Actinic Keratoses",
        "description": "Precancerous lesions (may progress to squamous cell carcinoma)",
        "urgency": "Schedule dermatologist appointment within 1-2 weeks",
    },
    "bcc": {
        "risk": "RED",
        "name": "Basal Cell Carcinoma",
        "description": "Most common form of skin cancer; slow-growing but locally invasive",
        "urgency": "Schedule dermatologist appointment within 1-2 weeks",
    },
    "bkl": {
        "risk": "YELLOW",
        "name": "Benign Keratosis",
        "description": "Non-cancerous growths including seborrheic keratoses and solar lentigo",
        "urgency": "Visit dermatologist within 1-2 months for confirmation",
    },
    "df": {
        "risk": "YELLOW",
        "name": "Dermatofibroma",
        "description": "Benign fibrous skin nodule; usually harmless",
        "urgency": "Visit dermatologist within 1-2 months if concerned",
    },
    "mel": {
        "risk": "RED",
        "name": "Melanoma",
        "description": "Most dangerous skin cancer; can metastasize if untreated",
        "urgency": "URGENT — seek immediate dermatology consultation",
    },
    "nv": {
        "risk": "GREEN",
        "name": "Melanocytic Nevi",
        "description": "Common benign moles; usually harmless",
        "urgency": "Continue regular self-examinations",
    },
    "vasc": {
        "risk": "YELLOW",
        "name": "Vascular Lesions",
        "description": "Blood vessel-related lesions (angiomas, angiokeratomas, etc.)",
        "urgency": "Visit dermatologist within 1-2 months if changes occur",
    },
}

RISK_COLORS = {
    "RED": "🔴",
    "YELLOW": "🟡",
    "GREEN": "🟢",
}

# Hospital guidance (India-focused example)
HOSPITAL_GUIDANCE = [
    "🏥 AIIMS — All India Institute of Medical Sciences, New Delhi",
    "🏥 Apollo Hospitals — Nationwide",
    "🏥 Fortis Healthcare — Nationwide",
    "🏥 Tata Memorial Hospital, Mumbai",
    "🏥 Regional Cancer Centers (RCCs) across India",
]
