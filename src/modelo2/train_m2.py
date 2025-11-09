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
import numpy as np
import datetime

def entrenar_modelo2(data_path=None, num_epochs=80):
    """
    Entrena Modelo 2 para regresión de calorías con reporte detallado
    """
    print(f"\n═══════════ 🟢 INICIANDO ENTRENAMIENTO MODELO 2 (CNN) 🟢 ═══════════ {datetime.datetime.now()}\n")
    # Ruta automática
    if data_path is None:
        if os.path.exists('/content/drive/MyDrive/DL_CaloriasComida/datos/originales/nutrition5k'):
            data_path = '/content/drive/MyDrive/DL_CaloriasComida/datos/originales/nutrition5k'
        elif os.path.exists('datos/originales/nutrition5k'):
            data_path = 'datos/originales/nutrition5k'
        else:
            raise ValueError('No se encontró Nutrition5k. Verifica la ruta.')

    os.makedirs('modelos', exist_ok=True)
    os.makedirs('resultados/modelo2', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Dispositivo: {device}")

    # Augmentations mejorados
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.RandomErasing(p=0.25),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("→ Cargando datasets y calculando estadísticas...")
    train_dataset = Nutrition5kDataset(data_path, split='train', transform=train_transform, normalize_target=True)
    val_dataset = Nutrition5kDataset(data_path, split='val', transform=val_transform, normalize_target=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    mean_cal = train_dataset.mean_cal
    std_cal = train_dataset.std_cal
    print(f"📊 Target stats: mean = {mean_cal:.2f} kcal | std = {std_cal:.2f} kcal")
    print(f"📦 Train: {len(train_loader.dataset)}   | Val: {len(val_loader.dataset)} muestras")

    modelo = CNNRegressor().to(device)
    total_params = sum(p.numel() for p in modelo.parameters())
    print(f"\n🧠 Modelo CNN creado. Total parámetros: {total_params:,}\n")

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(modelo.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=7, factor=0.3, min_lr=1e-5)
    patience = 20
    min_val_mae = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}

    for epoch in range(num_epochs):
        print(f"\n{'='*65}\n⏳ Época {epoch+1}/{num_epochs} ({datetime.datetime.now().strftime('%H:%M:%S')})")
        modelo.train()
        train_loss, train_mae = 0.0, 0.0
        pbar = tqdm(train_loader, ncols=70)
        for images, calories in pbar:
            images, calories = images.to(device), calories.to(device)
            optimizer.zero_grad()
            outputs = modelo(images)
            loss = criterion(outputs, calories)
            loss.backward()
            optimizer.step()
            # MAE denormalizado
            outputs_denorm = outputs * std_cal + mean_cal
            calories_denorm = calories * std_cal + mean_cal
            mae = torch.mean(torch.abs(outputs_denorm - calories_denorm)).item()
            train_loss += loss.item()
            train_mae += mae
            # Print parcial por lote (cada 100 batches)
            if pbar.n % 100 == 0 and pbar.n > 0:
                pbar.set_postfix({'Batch Loss': loss.item(), 'MAE': f'{mae:.1f}'})
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        print(f"  🏋️‍♂️ Train: Loss={train_loss:.4f} | MAE={train_mae:.2f} kcal")

        # Validación
        modelo.eval()
        val_loss, val_mae = 0.0, 0.0
        with torch.no_grad():
            vpbar = tqdm(val_loader, ncols=70)
            for images, calories in vpbar:
                images, calories = images.to(device), calories.to(device)
                outputs = modelo(images)
                loss = criterion(outputs, calories)
                outputs_denorm = outputs * std_cal + mean_cal
                calories_denorm = calories * std_cal + mean_cal
                mae = torch.mean(torch.abs(outputs_denorm - calories_denorm)).item()
                val_loss += loss.item()
                val_mae += mae
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        print(f"  🧪 Val:   Loss={val_loss:.4f} | MAE={val_mae:.2f} kcal")

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        # Early stopping y guardar mejor modelo
        if val_mae < min_val_mae - 1e-2:
            min_val_mae = val_mae
            epochs_no_improve = 0
            torch.save(modelo.state_dict(), 'modelos/modelo2_mejor.pth')
            print(f"  ✅ Mejor modelo guardado | Val MAE: {val_mae:.2f} kcal")
        else:
            epochs_no_improve += 1
            print(f"  ⏸️  Sin mejora ({epochs_no_improve}/{patience} épocas)")

        if epochs_no_improve >= patience:
            print("\n🛑 Early stopping (sin mejora)")
            break

    torch.save(modelo.state_dict(), 'modelos/modelo2_final.pth')
    with open('resultados/modelo2/train_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\n✅ ENTRENAMIENTO COMPLETADO | Mejor MAE: {min_val_mae:.2f} kcal (denormalizado)")
    print(f"📁 Modelos guardados en './modelos/'\n📈 Historia en './resultados/modelo2/'\n")

if __name__ == '__main__':
    entrenar_modelo2()
