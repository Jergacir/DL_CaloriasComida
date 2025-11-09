"""
Script de entrenamiento ResNet50 + Food-101
Ejecutar desde raíz del proyecto: python src/modelo1_resnet50/train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from dataset import Food101Dataset, prepare_food101_splits
from model import ResNet50Food101
import config

# ═══════════════════════════════════════════════════════════
# FUNCIONES DE ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════

def train_epoch(modelo, loader, criterion, optimizer, device):
    """Entrena una época"""
    modelo.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = modelo(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(loader):.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    return running_loss / len(loader), 100 * correct / total


def evaluate(modelo, loader, criterion, device):
    """Evalúa el modelo"""
    modelo.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Evaluating')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = modelo(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(loader):.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
    
    return running_loss / len(loader), 100 * correct / total


def plot_training_curves(history, save_path):
    """Genera gráficas de entrenamiento"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Época', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss durante Entrenamiento', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    mejor_acc = max(history['val_acc'])
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].axhline(y=mejor_acc, color='green', linestyle='--', 
                    label=f'Mejor: {mejor_acc:.2f}%')
    axes[1].set_xlabel('Época', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Accuracy durante Entrenamiento', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Gráficas guardadas en: {save_path}")

# ═══════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════

def main():
    """Entrenamiento completo ResNet50 + Food-101"""
    
    print("="*70)
    print("🚀 ENTRENAMIENTO RESNET50 + FOOD-101")
    print("="*70)
    
    # Device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"\n🔥 Dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # ═══════════════════════════════════════════════════════
    # 1. PREPARAR DATOS
    # ═══════════════════════════════════════════════════════
    
    print(f"\n📂 Cargando dataset desde: {config.DATA_PATH}")
    
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, classes = \
        prepare_food101_splits(
            config.DATA_PATH,
            train_split=config.TRAIN_SPLIT,
            val_split=config.VAL_SPLIT
        )
    
    # Guardar clases
    with open(config.CLASSES_FILE, 'w') as f:
        json.dump(classes, f, indent=4)
    print(f"\n✓ Clases guardadas en: {config.CLASSES_FILE}")
    
    # Transformaciones
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        normalize
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Datasets y Loaders
    train_dataset = Food101Dataset(train_paths, train_labels, transform=train_transform)
    val_dataset = Food101Dataset(val_paths, val_labels, transform=test_transform)
    test_dataset = Food101Dataset(test_paths, test_labels, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                              shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                           shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    
    print(f"\n✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches:   {len(val_loader)}")
    print(f"✓ Test batches:  {len(test_loader)}")
    
    # ═══════════════════════════════════════════════════════
    # 2. CREAR MODELO
    # ═══════════════════════════════════════════════════════
    
    print("\n🧠 Creando modelo ResNet50...")
    modelo = ResNet50Food101(num_classes=config.NUM_CLASSES, freeze_base=True).to(device)
    
    total_params, trainable_params = modelo.count_parameters()
    print(f"\n📊 Parámetros:")
    print(f"   Total: {total_params:,}")
    print(f"   Entrenables: {trainable_params:,}")
    print(f"   Congelados: {total_params - trainable_params:,}")
    
    # ═══════════════════════════════════════════════════════
    # 3. FASE 1: Entrenar solo FC
    # ═══════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("🚀 FASE 1: Entrenando capas FC (base congelada)")
    print("="*70)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(modelo.resnet.fc.parameters(), 
                           lr=config.LEARNING_RATE, 
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    mejor_acc = 0.0
    epocas_sin_mejora = 0
    
    for epoch in range(config.FASE1_EPOCHS):
        print(f"\n{'='*70}")
        print(f"Época {epoch+1}/{config.FASE1_EPOCHS} [FASE 1]")
        print(f"{'='*70}")
        
        train_loss, train_acc = train_epoch(modelo, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(modelo, val_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\n📊 Época {epoch+1}:")
        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        if val_acc > mejor_acc:
            mejor_acc = val_acc
            epocas_sin_mejora = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': modelo.state_dict(),
                'val_acc': val_acc,
                'classes': classes
            }, config.MODEL_FASE1)
            print(f"   ✅ Mejor modelo (Val Acc: {val_acc:.2f}%)")
        else:
            epocas_sin_mejora += 1
            print(f"   ⏸️  Sin mejora ({epocas_sin_mejora}/{config.FASE1_PATIENCE})")
        
        if epocas_sin_mejora >= config.FASE1_PATIENCE:
            print(f"\n🛑 EARLY STOPPING")
            break
    
    print(f"\n✅ FASE 1 COMPLETADA. Mejor: {mejor_acc:.2f}%")
    
    # ═══════════════════════════════════════════════════════
    # 4. FASE 2: Fine-tuning
    # ═══════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("🚀 FASE 2: Fine-tuning completo")
    print("="*70)
    
    modelo.unfreeze()
    optimizer = optim.AdamW(modelo.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=3)
    epocas_sin_mejora = 0
    
    for epoch in range(config.FASE2_EPOCHS):
        print(f"\n{'='*70}")
        print(f"Época {epoch+1}/{config.FASE2_EPOCHS} [FASE 2]")
        print(f"{'='*70}")
        
        train_loss, train_acc = train_epoch(modelo, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(modelo, val_loader, criterion, device)
        scheduler.step(val_acc)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\n📊 Época {epoch+1}:")
        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        if val_acc > mejor_acc:
            mejor_acc = val_acc
            epocas_sin_mejora = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': modelo.state_dict(),
                'val_acc': val_acc,
                'classes': classes
            }, config.MODEL_BEST)
            print(f"   ✅ Mejor modelo (Val Acc: {val_acc:.2f}%)")
        else:
            epocas_sin_mejora += 1
            print(f"   ⏸️  Sin mejora ({epocas_sin_mejora}/{config.FASE2_PATIENCE})")
        
        if epocas_sin_mejora >= config.FASE2_PATIENCE:
            print(f"\n🛑 EARLY STOPPING")
            break
    
    # Guardar final
    torch.save({'model_state_dict': modelo.state_dict(), 'classes': classes}, 
               config.MODEL_FINAL)
    
    # ═══════════════════════════════════════════════════════
    # 5. EVALUACIÓN FINAL Y RESULTADOS
    # ═══════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("🎯 EVALUACIÓN FINAL")
    print("="*70)
    
    # Cargar mejor modelo
    checkpoint = torch.load(config.MODEL_BEST)
    modelo.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluar en test
    test_loss, test_acc = evaluate(modelo, test_loader, criterion, device)
    
    print(f"\n📊 RESULTADOS FINALES:")
    print(f"   Mejor Val Acc:  {mejor_acc:.2f}%")
    print(f"   Test Acc:       {test_acc:.2f}%")
    print(f"   Diferencia:     {abs(test_acc - mejor_acc):.2f}%")
    
    # Guardar historial
    with open(config.HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\n✓ Historial: {config.HISTORY_FILE}")
    
    # Gráficas
    plot_training_curves(history, config.CURVES_FILE)
    
    print(f"\n✅ ENTRENAMIENTO COMPLETADO")
    print(f"   Modelos en: {config.MODELS_PATH}")
    print(f"   Resultados en: {config.RESULTS_PATH}")


if __name__ == '__main__':
    main()
