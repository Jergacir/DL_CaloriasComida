"""
Modelo ResNet50 para Food-101
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class ResNet50Food101(nn.Module):
    """ResNet50 preentrenado para Food-101"""
    
    def __init__(self, num_classes=101, freeze_base=True):
        super(ResNet50Food101, self).__init__()
        
        # Cargar ResNet50 preentrenado
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Congelar capas base
        if freeze_base:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Reemplazar última capa
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
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
