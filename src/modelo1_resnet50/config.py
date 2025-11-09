"""
Configuración para ResNet50 + Food-101
"""

import os

# ═══════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'datos', 'food101')
MODELS_PATH = os.path.join(BASE_DIR, 'modelos', 'food101')
RESULTS_PATH = os.path.join(BASE_DIR, 'resultados', 'food101')

# Crear directorios
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════

# Data
NUM_CLASSES = 101
IMG_SIZE = 224
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Training
BATCH_SIZE = 64          # Ajustar según GPU
NUM_WORKERS = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Epochs y Early Stopping
FASE1_EPOCHS = 50
FASE2_EPOCHS = 50
FASE1_PATIENCE = 7
FASE2_PATIENCE = 10

# Device
DEVICE = 'cuda'  # o 'cpu'

# ═══════════════════════════════════════════════════════════
# MODEL SAVE PATHS
# ═══════════════════════════════════════════════════════════

MODEL_FASE1 = os.path.join(MODELS_PATH, 'resnet50_food101_fase1.pth')
MODEL_BEST = os.path.join(MODELS_PATH, 'resnet50_food101_best.pth')
MODEL_FINAL = os.path.join(MODELS_PATH, 'resnet50_food101_final.pth')
CLASSES_FILE = os.path.join(MODELS_PATH, 'food101_classes.json')
HISTORY_FILE = os.path.join(RESULTS_PATH, 'training_history.json')
CURVES_FILE = os.path.join(RESULTS_PATH, 'training_curves.png')
