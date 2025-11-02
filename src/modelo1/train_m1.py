# src/modelo1/train_m1.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json
import os

from dataset import Food11Dataset
from cnn_clasificador import CNNClasificador

def entrenar_modelo1(data_path='datos/originales/food11', num_epochs=30):
    """Entrena el Modelo 1"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    # Transformaciones
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets y DataLoaders
    train_dataset = Food11Dataset(data_path, 'training', train_transform)
    val_dataset = Food11Dataset(data_path, 'validation', val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Modelo
    modelo = CNNClasificador(num_clases=11).to(device)
    
    # Optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelo.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Entrenamiento
    mejor_val_acc = 0.0
    historial = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n{'='*70}")
    print(f"ENTRENANDO MODELO 1 - {num_epochs} ÉPOCAS")
    print(f"{'='*70}\n")
    
    for epoch in range(num_epochs):
        # TRAIN
        modelo.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Época {epoch+1}/{num_epochs} [TRAIN]'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = modelo(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # VALIDATION
        modelo.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Época {epoch+1}/{num_epochs} [VAL]'):
                images, labels = images.to(device), labels.to(device)
                outputs = modelo(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        scheduler.step(val_loss)
        
        # Guardar historial
        historial['train_loss'].append(train_loss)
        historial['train_acc'].append(train_acc)
        historial['val_loss'].append(val_loss)
        historial['val_acc'].append(val_acc)
        
        # Guardar mejor modelo
        if val_acc > mejor_val_acc:
            mejor_val_acc = val_acc
            os.makedirs('modelos', exist_ok=True)
            torch.save(modelo.state_dict(), 'modelos/modelo1_clasificador.pth')
        
        print(f"\nÉpoca {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}")
    
    # Guardar historial
    os.makedirs('resultados/modelo1', exist_ok=True)
    with open('resultados/modelo1/historial.json', 'w') as f:
        json.dump(historial, f)
    
    print(f"\n{'='*70}")
    print(f"ENTRENAMIENTO COMPLETADO")
    print(f"Mejor Val Accuracy: {mejor_val_acc:.4f}")
    print(f"{'='*70}")

if __name__ == '__main__':
    entrenar_modelo1()
