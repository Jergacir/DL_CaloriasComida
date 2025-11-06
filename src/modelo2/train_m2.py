# ═══════════════════════════════════════════════════════════
# src/modelo2/train_m2.py
# Entrena CNN para regresión de calorías (Nutrition5k)
# ═══════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json
import os

from dataset_nutrition5k import Nutrition5kDataset
from cnn_regressor import CNNRegressor


def entrenar_modelo2(data_path=None, num_epochs=50):
    """
    Entrena Modelo 2 para regresión de calorías
    
    Args:
        data_path: Path a nutrition5k/
        num_epochs: Número de épocas (default: 50)
    """
    
    # ═════════════════════════════════════════════════════════════
    # AUTODETECTAR RUTA
    # ═════════════════════════════════════════════════════════════
    
    if data_path is None:
        # Prioridad 1: Colab Drive
        if os.path.exists('/content/drive/MyDrive/DL_CaloriasComida/datos/originales/nutrition5k'):
            data_path = '/content/drive/MyDrive/DL_CaloriasComida/datos/originales/nutrition5k'
        # Prioridad 2: Local
        elif os.path.exists('datos/originales/nutrition5k'):
            data_path = 'datos/originales/nutrition5k'
        else:
            raise ValueError("No se encontró Nutrition5k. Verifica la ruta.")
    
    # ═════════════════════════════════════════════════════════════
    # CREAR CARPETAS
    # ═════════════════════════════════════════════════════════════
    
    os.makedirs('modelos', exist_ok=True)
    os.makedirs('resultados/modelo2', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    print(f"📂 Datos desde: {data_path}\n")
    
    # ═════════════════════════════════════════════════════════════
    # TRANSFORMACIONES
    # ═════════════════════════════════════════════════════════════
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
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
    
    print("📊 Cargando datasets...")
    train_dataset = Nutrition5kDataset(data_path, split='train', transform=train_transform, train_ratio=0.8)
    val_dataset = Nutrition5kDataset(data_path, split='val', transform=val_transform, train_ratio=0.8)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}\n")
    
    # ═════════════════════════════════════════════════════════════
    # MODELO
    # ═════════════════════════════════════════════════════════════
    
    print("🧠 Inicializando modelo CNN Regressor...")
    modelo = CNNRegressor().to(device)
    total_params = sum(p.numel() for p in modelo.parameters())
    print(f"   Parámetros totales: {total_params:,}\n")
    
    # ═════════════════════════════════════════════════════════════
    # LOSS Y OPTIMIZADOR
    # ═════════════════════════════════════════════════════════════
    
    # Para regresión: MSE Loss (Mean Squared Error)
    criterion = nn.MSELoss()
    
    # Optimizador
    optimizer = optim.Adam(modelo.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # ═════════════════════════════════════════════════════════════
    # ENTRENAMIENTO
    # ═════════════════════════════════════════════════════════════
    
    mejor_val_mae = float('inf')
    sin_mejora = 0
    patience = 10
    historial = {
        'train_loss': [], 'train_mae': [],
        'val_loss': [], 'val_mae': []
    }
    
    print(f"{'='*70}")
    print(f"🏋️ ENTRENANDO MODELO 2 - REGRESIÓN DE CALORÍAS - {num_epochs} ÉPOCAS")
    print(f"{'='*70}")
    print(f"Configuración:")
    print(f"  ✓ Loss: MSE (Mean Squared Error)")
    print(f"  ✓ Métrica: MAE (Mean Absolute Error)")
    print(f"  ✓ Batch size: 32")
    print(f"  ✓ Optimizer: Adam (lr=0.001)")
    print(f"  ✓ Early stopping: {patience} épocas")
    print(f"{'='*70}\n")
    
    for epoch in range(num_epochs):
        # ═════════════════════════════════════════════════════════
        # TRAIN
        # ═════════════════════════════════════════════════════════
        
        modelo.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for images, calories in tqdm(train_loader, desc=f'Época {epoch+1}/{num_epochs} [TRAIN]', leave=False):
            images, calories = images.to(device), calories.to(device)
            
            # Forward
            outputs = modelo(images)
            loss = criterion(outputs, calories)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Métricas
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(outputs - calories)).item()
        
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        
        # ═════════════════════════════════════════════════════════
        # VALIDATION
        # ═════════════════════════════════════════════════════════
        
        modelo.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for images, calories in tqdm(val_loader, desc=f'Época {epoch+1}/{num_epochs} [VAL]', leave=False):
                images, calories = images.to(device), calories.to(device)
                
                outputs = modelo(images)
                loss = criterion(outputs, calories)
                
                val_loss += loss.item()
                val_mae += torch.mean(torch.abs(outputs - calories)).item()
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Guardar historial
        historial['train_loss'].append(train_loss)
        historial['train_mae'].append(train_mae)
        historial['val_loss'].append(val_loss)
        historial['val_mae'].append(val_mae)
        
        # Early stopping
        if val_mae < mejor_val_mae:
            mejor_val_mae = val_mae
            sin_mejora = 0
            torch.save({
                'model_state_dict': modelo.state_dict(),
                'epoch': epoch,
                'val_mae': val_mae
            }, 'modelos/modelo2_mejor.pth')
            mejor_str = "✓ Mejor modelo guardado"
        else:
            sin_mejora += 1
            mejor_str = ""
        
        # MOSTRAR RESULTADOS
        print(f"\nÉpoca {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.2f} kcal")
        print(f"  Val Loss: {val_loss:.4f}   | Val MAE: {val_mae:.2f} kcal")
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
    
    with open('resultados/modelo2/historial.json', 'w') as f:
        json.dump(historial, f, indent=4)
    
    torch.save(modelo.state_dict(), 'modelos/modelo2_final.pth')
    
    print(f"\n{'='*70}")
    print(f"✅ ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"\n📊 RESULTADOS FINALES:")
    print(f"   Mejor Val MAE: {mejor_val_mae:.2f} kcal")
    print(f"   (Menor es mejor)")
    print(f"\n💾 ARCHIVOS GENERADOS:")
    print(f"   ✓ modelos/modelo2_mejor.pth")
    print(f"   ✓ modelos/modelo2_final.pth")
    print(f"   ✓ resultados/modelo2/historial.json")
    print(f"{'='*70}")


if __name__ == '__main__':
    entrenar_modelo2(num_epochs=50)
