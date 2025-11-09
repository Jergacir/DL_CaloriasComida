"""
Dataset loader para Food-101
"""

import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Food101Dataset(Dataset):
    """Dataset para Food-101"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: Lista de paths a imágenes
            labels: Lista de labels correspondientes
            transform: Transformaciones de torchvision
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Cargar imagen
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        
        return image, label


def prepare_food101_splits(data_path, train_split=0.70, val_split=0.15, random_state=42):
    """
    Prepara splits de Food-101 (train/val/test)
    
    Args:
        data_path: Path base del dataset
        train_split: Proporción de entrenamiento (default: 0.70)
        val_split: Proporción de validación (default: 0.15)
        random_state: Seed para reproducibilidad
        
    Returns:
        train_paths, train_labels, val_paths, val_labels, 
        test_paths, test_labels, classes
    """
    images_path = os.path.join(data_path, 'images')
    
    # Obtener clases
    classes = sorted([d for d in os.listdir(images_path) 
                     if os.path.isdir(os.path.join(images_path, d))])
    
    print(f"✓ Clases encontradas: {len(classes)}")
    
    # Mapeo clase -> índice
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Recolectar paths e labels
    all_image_paths = []
    all_labels = []
    
    print("\n📊 Escaneando imágenes...")
    for cls in tqdm(classes):
        cls_path = os.path.join(images_path, cls)
        images = glob.glob(os.path.join(cls_path, '*.jpg'))
        
        all_image_paths.extend(images)
        all_labels.extend([class_to_idx[cls]] * len(images))
    
    print(f"✓ Total imágenes: {len(all_image_paths):,}")
    
    # Split 70/15/15
    print(f"\n🔀 Dividiendo dataset ({int(train_split*100)}%/{int(val_split*100)}%/{int((1-train_split-val_split)*100)}%)...")
    
    # Train + Temp (val+test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_image_paths, 
        all_labels,
        test_size=(1 - train_split),
        random_state=random_state,
        stratify=all_labels
    )
    
    # Val + Test
    val_test_split = val_split / (1 - train_split)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=(1 - val_test_split),
        random_state=random_state,
        stratify=temp_labels
    )
    
    # Verificar
    total = len(all_image_paths)
    print(f"\n📊 Distribución final:")
    print(f"   Train: {len(train_paths):,} ({100*len(train_paths)/total:.1f}%)")
    print(f"   Val:   {len(val_paths):,} ({100*len(val_paths)/total:.1f}%)")
    print(f"   Test:  {len(test_paths):,} ({100*len(test_paths)/total:.1f}%)")
    
    return (train_paths, train_labels, 
            val_paths, val_labels, 
            test_paths, test_labels, 
            classes)
