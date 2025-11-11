import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class Nutrition5kDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.metadata = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        dish_id = self.metadata.iloc[idx]['dish_id']
        img_path = os.path.join(self.img_dir, dish_id, 'rgb.png')
        
        # Cargar imagen
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Etiquetas
        calorias = self.metadata.iloc[idx]['total_calories']
        masa = self.metadata.iloc[idx]['total_mass']
        
        return {
            'imagen': image,
            'calorias': torch.tensor(calorias, dtype=torch.float32),
            'masa': torch.tensor(masa, dtype=torch.float32)
        }

def obtener_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
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
    
    return train_transform, val_transform

def crear_dataloaders(config):
    train_transform, val_transform = obtener_transforms()
    
    # Cargar datasets
    train_dataset = Nutrition5kDataset(
        csv_path=os.path.join(config.NUTRITION5K_DIR, 'metadata/dish_metadata_cafe1.csv'),
        img_dir=os.path.join(config.NUTRITION5K_DIR, 'imagery/realsense_overhead'),
        transform=train_transform
    )
    
    val_dataset = Nutrition5kDataset(
        csv_path=os.path.join(config.NUTRITION5K_DIR, 'metadata/dish_metadata_cafe2.csv'),
        img_dir=os.path.join(config.NUTRITION5K_DIR, 'imagery/realsense_overhead'),
        transform=val_transform
    )
    
    # Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                           shuffle=False, num_workers=config.NUM_WORKERS)
    
    return train_loader, val_loader
