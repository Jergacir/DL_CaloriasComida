from src.configuracion import Config
from src.cargador_datos import crear_dataloaders
from src.modelos import CNNSimple
from src.entrenar import entrenar_modelo
import torch

def main():
    print("\n" + "="*60)
    print("PROYECTO: Estimación de Calorías con Deep Learning")
    print("="*60 + "\n")
    
    # Configuración
    config = Config()
    print(f"Dispositivo: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Épocas: {config.EPOCAS}\n")
    
    # Cargar datos
    print("Cargando datos...")
    train_loader, val_loader = crear_dataloaders(config)
    print(f"✓ Datos cargados")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}\n")
    
    # Crear modelo
    print("Creando modelo...")
    modelo = CNNSimple(num_salidas=2)
    total_params = sum(p.numel() for p in modelo.parameters())
    print(f"✓ Modelo creado")
    print(f"  - Total parámetros: {total_params:,}\n")
    
    # Entrenar
    print("Iniciando entrenamiento...")
    historial = entrenar_modelo(modelo, train_loader, val_loader, config)
    
    print("\n" + "="*60)
    print("✓ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"Mejor val_loss: {min(historial['val_loss']):.4f}")
    print(f"Modelo guardado en: {config.CHECKPOINT_DIR}/mejor_modelo.pth")

if __name__ == '__main__':
    main()
