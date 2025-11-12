# cnn_regressor.py
import torch.nn as nn

class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), # Capa convolucional
            nn.BatchNorm2d(64), # Normalización
            nn.ReLU(), # Activación
            nn.MaxPool2d(2), # Reducción de tamaño

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*14*14, 512), # Reduce a 512 neuronas
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1) # Salida: 1 número (calorías)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Aplana 14×14×512 → 100,352
        x = self.regressor(x)
        return x.squeeze()
