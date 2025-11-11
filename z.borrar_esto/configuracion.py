import torch

class Config:
    # Rutas de datos
    NUTRITION5K_DIR = 'datos/originales/nutrition5k'
    FOOD101_DIR = 'datos/originales/food101'
    PROCESADOS_DIR = 'datos/procesados'
    
    # Parámetros de datos
    TAMANO_IMG = (224, 224)
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Parámetros del modelo
    NUM_CLASES = 101  # Para Food-101
    USAR_PROFUNDIDAD = True
    
    # Parámetros de entrenamiento
    EPOCAS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    
    # Hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Rutas de guardado
    CHECKPOINT_DIR = 'modelos_guardados/checkpoints'
    RESULTADOS_DIR = 'resultados'
    
    # División de datos
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
