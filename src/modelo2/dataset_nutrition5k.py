# ═══════════════════════════════════════════════════════════
# src/modelo2/dataset_nutrition5k.py 
# ═══════════════════════════════════════════════════════════

import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import io
from PIL import Image

class Nutrition5kDataset(Dataset):
    """
    Dataset para Nutrition5k (Regresión de Calorías)
    Las imágenes están en formato bytes dentro del DataFrame
    """
    
    def __init__(self, data_path, split='train', transform=None, train_ratio=0.8):
        """
        Args:
            data_path: Path a nutrition5k/
            split: 'train' o 'val'
            transform: Transformaciones
            train_ratio: Proporción train/val
        """
        # Cargar metadata (dishes.xlsx)
        dishes_path = f"{data_path}/dishes.xlsx"
        self.dishes = pd.read_excel(dishes_path)
        
        # Cargar imágenes (dish_images.pkl)
        images_path = f"{data_path}/dish_images.pkl"
        with open(images_path, 'rb') as f:
            self.images_df = pickle.load(f)
        
        # Crear mapeo dish_id -> calorías
        self.calorie_map = dict(zip(self.dishes['dish_id'], self.dishes['total_calories']))
        
        # Filtrar solo platos que tienen imagen Y calorías
        self.images_df = self.images_df[self.images_df['dish'].isin(self.calorie_map.keys())].reset_index(drop=True)
        
        print(f"⚠️ Dataset: {len(self.images_df)} platos con imagen y calorías")
        
        # Split train/val
        n_total = len(self.images_df)
        n_train = int(n_total * train_ratio)
        
        if split == 'train':
            self.images_df = self.images_df.iloc[:n_train].reset_index(drop=True)
        else:
            self.images_df = self.images_df.iloc[n_train:].reset_index(drop=True)
        
        self.transform = transform
        
        # Calcular estadísticas de calorías
        calories_list = [self.calorie_map[dish_id] for dish_id in self.images_df['dish']]
        
        print(f"✓ Nutrition5k {split.upper()} cargado")
        print(f"  - Samples: {len(self.images_df)}")
        print(f"  - Rango calorías: {min(calories_list):.0f} - {max(calories_list):.0f} kcal")
    
    def __len__(self):
        return len(self.images_df)
    
    def __getitem__(self, idx):
        # Obtener dish_id
        dish_id = self.images_df.iloc[idx]['dish']
        
        # Obtener calorías
        calories = self.calorie_map[dish_id]
        
        # Decodificar imagen RGB desde bytes
        img_bytes = self.images_df.iloc[idx]['rgb_image']
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(calories, dtype=torch.float32)
