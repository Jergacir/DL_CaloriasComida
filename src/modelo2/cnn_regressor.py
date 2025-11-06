# ═══════════════════════════════════════════════════════════
# src/modelo2/cnn_regressor.py
# CNN para regresión de calorías
# ═══════════════════════════════════════════════════════════

import torch
import torch.nn as nn

class CNNRegressor(nn.Module):
    """
    CNN para estimar calorías (regresión)
    Input: Imagen (3, 224, 224)
    Output: Calorías (valor escalar)
    """
    
    def __init__(self):
        super(CNNRegressor, self).__init__()
        
        # Feature extractor (similar a Modelo 1)
        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Bloque 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Bloque 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Bloque 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Regressor (Output: 1 valor - calorías)
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # ← OUTPUT: 1 neurona (calorías)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x.squeeze()  # Shape: (batch_size,) en lugar de (batch_size, 1)
