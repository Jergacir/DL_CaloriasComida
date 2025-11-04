# ═══════════════════════════════════════════════════════════
# src/modelo1/train_m1_mejorado.py
# Mismo modelo pero con cambios inteligentes
# ═══════════════════════════════════════════════════════════

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

def entrenar_modelo1_mejorado(data_path=None, num_epochs=80):
    """Entrena con más épocas y cambios de hiperparámetros"""
    
    # AUTODETECTAR RUTA
    if data_path is None:
        # Prioridad 1: Colab
        if os.path.exists('/content/drive/MyDrive/DL_CaloriasComida/datos/originales/food11'):
            data_path = '/content/drive/MyDrive/DL_CaloriasComida/datos/originales/food11'
        # Prioridad 2: Local
        elif os.path.exists('datos/originales/food11'):
            data_path = 'datos/originales/food11'
        else:
            raise ValueError("No se encontró Food-11. Verifica la ruta.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    print(f"📂 Datos desde: {data_path}\n")
    
    # CAMBIO 1: Aumentar batch size (mejor gradientes)
    # Transforms MÁS AGRESIVOS
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),           # ← NUEVO
        transforms.RandomRotation(45),                  # ← Aumentado de 15
        transforms.ColorJitter(
            brightness=0.4,                             # ← Aumentado
            contrast=0.4,                               # ← Aumentado
            saturation=0.3,                             # ← NUEVO
            hue=0.1                                     # ← NUEVO
        ),
        transforms.RandomAffine(
            degrees=20, 
            translate=(0.15, 0.15),                     # ← NUEVO
            scale=(0.7, 1.3)                            # ← NUEVO
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # ← NUEVO
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
    
    # CAMBIO 2: Aumentar batch size (de 32 a 64)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Modelo (igual, pero entrenamiento diferente)
    modelo = CNNClasificador(num_clases=11).to(device)
    
    # CAMBIO 3: Optimizador mejorado
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # ← Label smoothing
    optimizer = optim.AdamW(
        modelo.parameters(), 
        lr=0.001,
        weight_decay=1e-4,                              # ← Regularización L2
        betas=(0.9, 0.999)
    )
    
    # CAMBIO 4: Mejor scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=20,      # Período inicial
        T_mult=1,    # Multiplicador de período
        eta_min=1e-6 # Learning rate mínimo
    )
    
    # Entrenamiento
    mejor_val_acc = 0.0
    sin_mejora = 0
    patience = 15  # Early stopping
    historial = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n{'='*70}")
    print(f"🏋️ ENTRENANDO MODELO 1 MEJORADO - {num_epochs} ÉPOCAS")
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
            
            # CAMBIO 5: Gradient clipping (evita explosión de gradientes)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            
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
        
        # Scheduler
        scheduler.step()
        
        # Guardar historial
        historial['train_loss'].append(train_loss)
        historial['train_acc'].append(train_acc)
        historial['val_loss'].append(val_loss)
        historial['val_acc'].append(val_acc)
        
        # Early stopping
        if val_acc > mejor_val_acc:
            mejor_val_acc = val_acc
            sin_mejora = 0
            torch.save({
                'model_state_dict': modelo.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }, 'modelos/modelo1_mejorado.pth')
            print(f"✓ Mejor modelo guardado (val_acc: {val_acc:.4f})")
        else:
            sin_mejora += 1
        
        # Mostrar info
        print(f"\nÉpoca {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f} | Sin mejora: {sin_mejora}/{patience}")
        
        # Early stopping si no mejora
        if sin_mejora >= patience:
            print(f"\n⚠️ Early stopping en época {epoch+1}")
            break
    
    # Guardar historial
    os.makedirs('resultados/modelo1_mejorado', exist_ok=True)
    with open('resultados/modelo1_mejorado/historial.json', 'w') as f:
        json.dump(historial, f)
    
    print(f"\n{'='*70}")
    print(f"✅ ENTRENAMIENTO COMPLETADO")
    print(f"Mejor Val Accuracy: {mejor_val_acc:.4f} ({100*mejor_val_acc:.2f}%)")
    print(f"Comparado con versión anterior: +{100*(mejor_val_acc - 0.564):.2f}%")
    print(f"{'='*70}")

if __name__ == '__main__':
    entrenar_modelo1_mejorado(num_epochs=100)  # O 80 para empezar
