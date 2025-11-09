import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import io
from PIL import Image
import numpy as np
from torchvision import transforms

class Nutrition5kDataset(Dataset):
    def __init__(self, data_path, split='train', transform=None, train_ratio=0.8, normalize_target=False):
        dishes_path = f"{data_path}/dishes.xlsx"
        self.dishes = pd.read_excel(dishes_path)

        images_path = f"{data_path}/dish_images.pkl"
        with open(images_path, 'rb') as f:
            self.images_df = pickle.load(f)

        self.calorie_map = dict(zip(self.dishes['dish_id'], self.dishes['total_calories']))

        self.images_df = self.images_df[self.images_df['dish'].isin(self.calorie_map.keys())].reset_index(drop=True)

        n_total = len(self.images_df)
        n_train = int(n_total * train_ratio)

        if split == 'train':
            self.images_df = self.images_df.iloc[:n_train]
        elif split == 'val':
            self.images_df = self.images_df.iloc[n_train:]
        else:
            raise ValueError('split debe ser train o val')

        self.transform = transform
        self.normalize_target = normalize_target
        self.calories_list = [self.calorie_map[dish_id] for dish_id in self.images_df['dish']]

        if self.normalize_target:
            self.mean_cal = np.mean(self.calories_list)
            self.std_cal = np.std(self.calories_list)

        # Agrega una transformación para convertir PIL a Tensor si no viene con transform
        if self.transform is None:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, idx):
        row = self.images_df.iloc[idx]
        dish_id = row['dish']
        img_bytes = row['rgb_image']
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        calories = self.calorie_map[dish_id]
        if self.normalize_target:
            calories = (calories - self.mean_cal) / self.std_cal
        return image, torch.tensor(calories, dtype=torch.float32)
