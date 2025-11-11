import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_nutrition5k import Nutrition5kDataset
from cnn_regressor import CNNRegressor

def test_model(data_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_dataset = Nutrition5kDataset(data_path, split='test', transform=test_transform, normalize_target=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = CNNRegressor()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    mean_cal = test_dataset.mean_cal
    std_cal = test_dataset.std_cal

    total_mae = 0.0
    count = 0
    with torch.no_grad():
        for images, calories in test_loader:
            images, calories = images.to(device), calories.to(device)
            outputs = model(images)
            # Denormalizar para kcal reales
            outputs_denorm = outputs * std_cal + mean_cal
            calories_denorm = calories * std_cal + mean_cal
            mae = torch.mean(torch.abs(outputs_denorm - calories_denorm)).item()
            total_mae += mae
            count += 1

    print(f"MAE en test set: {total_mae / count:.2f} kcal")


if __name__ == '__main__':
    # Actualiza la ruta según tu carpeta de datos y el archivo del modelo guardado
    data_path = '/content/drive/MyDrive/DL_CaloriasComida/datos/originales/nutrition5k'
    model_path = '/content/drive/MyDrive/DL_CaloriasComida/modelos/modelo2_mejor.pth'
    test_model(data_path, model_path)