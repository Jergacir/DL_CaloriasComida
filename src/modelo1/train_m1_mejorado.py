# ═══════════════════════════════════════════════════════════
# src/modelo1/train_m1_mejorado.py
# Modelo 1 Mejorado: 80 épocas + mejoras + output completo
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
    """
    Entrena Modelo 1 con mejoras:
    - 80 épocas (con early stopping)
    - Data augmentation agresivo
    - Batch size: 64
    - Optimizador: AdamW
    - Scheduler: CosineAnnealing
    - Gradient clipping
    - Label smoothing
    """
    
    # ═════════════════════════════════════════════════════════════
    # AUTODETECTAR RUTA (Colab o Local)
    # ═════════════════════════════════════════════════════════════
    
    if data_path is None:
        # Prioridad 1: Colab
        if os.path.exists('/content/drive/MyDrive/DL_CaloriasComida/datos/originales/food11'):
            data_path = '/content/drive/MyDrive/DL_CaloriasComida/datos/originales/food11'
        # Prioridad 2: Local
        elif os.path.exists('datos/originales/food11'):
            data_path = 'datos/originales/food11'
        else:
            raise ValueError("No se encontró Food-11. Verifica la ruta.")
    
    # ═════════════════════════════════════════════════════════════
    # CREAR CARPETAS NECESARIAS
    # ═════════════════════════════════════════════════════════════
    
    os.makedirs('modelos', exist_ok=True)
    os.makedirs('resultados/modelo1_mejorado', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    print(f"📂 Datos desde: {data_path}\n")
    
    # ═════════════════════════════════════════════════════════════
    # TRANSFORMACIONES (Data Augmentation Agresivo)
    # ═════════════════════════════════════════════════════════════
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(45),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=20, 
            translate=(0.15, 0.15),
            scale=(0.7, 1.3)
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
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
    
    # ═════════════════════════════════════════════════════════════
    # CARGAR DATASETS
    # ═════════════════════════════════════════════════════════════
    
    train_dataset = Food11Dataset(data_path, 'training', train_transform)
    val_dataset = Food11Dataset(data_path, 'validation', val_transform)
    
    # DataLoaders con batch_size=64
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}\n")
    
    # ═════════════════════════════════════════════════════════════
    # MODELO Y OPTIMIZADOR
    # ═════════════════════════════════════════════════════════════
    
    modelo = CNNClasificador(num_clases=11).to(device)
    total_params = sum(p.numel() for p in modelo.parameters())
    print(f"🧠 Modelo: CNN Clasificador")
    print(f"   Parámetros totales: {total_params:,}\n")
    
    # Optimizador mejorado
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        modelo.parameters(), 
        lr=0.001,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Scheduler mejorado
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=20,
        T_mult=1,
        eta_min=1e-6
    )
    
    # ═════════════════════════════════════════════════════════════
    # ENTRENAMIENTO
    # ═════════════════════════════════════════════════════════════
    
    mejor_val_acc = 0.0
    sin_mejora = 0
    patience = 15  # Early stopping
    historial = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"{'='*70}")
    print(f"🏋️ ENTRENANDO MODELO 1 MEJORADO - {num_epochs} ÉPOCAS")
    print(f"{'='*70}")
    print(f"Mejoras aplicadas:")
    print(f"  ✓ Data Augmentation agresivo")
    print(f"  ✓ Batch size: 64")
    print(f"  ✓ AdamW optimizer")
    print(f"  ✓ Label smoothing")
    print(f"  ✓ CosineAnnealing scheduler")
    print(f"  ✓ Gradient clipping")
    print(f"  ✓ Early stopping")
    print(f"{'='*70}\n")
    
    for epoch in range(num_epochs):
        # TRAIN
        modelo.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Época {epoch+1}/{num_epochs} [TRAIN]', leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = modelo(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
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
            for images, labels in tqdm(val_loader, desc=f'Época {epoch+1}/{num_epochs} [VAL]', leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = modelo(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
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
            mejor_str = "✓ Mejor modelo guardado"
        else:
            sin_mejora += 1
            mejor_str = ""
        
        # MOSTRAR CADA ÉPOCA (como el original)
        print(f"\nÉpoca {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ({100*train_acc:.2f}%)")
        print(f"  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f} ({100*val_acc:.2f}%)")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f} | Sin mejora: {sin_mejora}/{patience}")
        if mejor_str:
            print(f"  {mejor_str}")
        
        # Early stopping
        if sin_mejora >= patience:
            print(f"\n⚠️ Early stopping en época {epoch+1}")
            break
    
    # ═════════════════════════════════════════════════════════════
    # GUARDAR RESULTADOS
    # ═════════════════════════════════════════════════════════════
    
    with open('resultados/modelo1_mejorado/historial.json', 'w') as f:
        json.dump(historial, f, indent=4)
    
    torch.save(modelo.state_dict(), 'modelos/modelo1_mejorado_final.pth')
    
    print(f"\n{'='*70}")
    print(f"✅ ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"\n📊 RESULTADOS FINALES:")
    print(f"   Mejor Val Accuracy (v2): {mejor_val_acc:.4f} ({100*mejor_val_acc:.2f}%)")
    print(f"   Versión anterior (v1):   0.5829 (58.29%)")
    print(f"   Mejora:                  +{100*(mejor_val_acc - 0.5829):.2f}%")
    print(f"\n💾 ARCHIVOS GENERADOS:")
    print(f"   ✓ modelos/modelo1_mejorado.pth")
    print(f"   ✓ modelos/modelo1_mejorado_final.pth")
    print(f"   ✓ resultados/modelo1_mejorado/historial.json")
    print(f"{'='*70}")


if __name__ == '__main__':
    entrenar_modelo1_mejorado(num_epochs=80)
