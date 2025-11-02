# src/modelo1/dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class Food11Dataset(Dataset):
    """Dataset para Food-11"""
    def __init__(self, root_dir, split='training', transform=None):
        self.root = os.path.join(root_dir, split)
        self.transform = transform
        
        self.clases = [
            'Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food',
            'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 
            'Vegetable-Fruit'
        ]
        self.clase_to_idx = {c: i for i, c in enumerate(self.clases)}
        
        # Recolectar imágenes
        self.samples = []
        for clase in self.clases:
            clase_dir = os.path.join(self.root, clase)
            if not os.path.exists(clase_dir):
                continue
            for img in os.listdir(clase_dir):
                if img.endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((
                        os.path.join(clase_dir, img),
                        self.clase_to_idx[clase]
                    ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
