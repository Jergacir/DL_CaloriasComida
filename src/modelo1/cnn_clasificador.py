# src/modelo1/cnn_clasificador.py

# import torch.nn as nn

# class CNNClasificador(nn.Module):
#     """CNN para clasificar 11 tipos de alimentos"""
#     def __init__(self, num_clases=11):
#         super(CNNClasificador, self).__init__()
        
#         self.features = nn.Sequential(
#             # Bloque 1
#             nn.Conv2d(3, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             # Bloque 2
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             # Bloque 3
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             # Bloque 4
#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
        
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(256 * 14 * 14, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, num_clases)
#         )
    
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

# ═══════════════════════════════════════════════════════════
# src/modelo1/cnn_clasificador.py
# Modelo ResNet50 para Food-101
# ═══════════════════════════════════════════════════════════

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class CNNClasificador(nn.Module):
    """ResNet50 preentrenado para Food-101"""
    
    def __init__(self, num_clases=101, freeze_base=True):
        super(CNNClasificador, self).__init__()
        
        # Cargar ResNet50 preentrenado
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Congelar capas base
        if freeze_base:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Reemplazar última capa
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),  # ← Valor mejorado (cambia a 0.6 si entrenaste con ese valor)
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),  # ← Valor mejorado (cambia a 0.5 si entrenaste con ese valor)
            nn.Linear(512, num_clases)
        )
    
    def forward(self, x):
        return self.resnet(x)
    
    def unfreeze(self):
        """Descongela todas las capas para fine-tuning"""
        for param in self.resnet.parameters():
            param.requires_grad = True
    
    def count_parameters(self):
        """Cuenta parámetros totales y entrenables"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# Alias para compatibilidad con código anterior
ResNet50Food101 = CNNClasificador


