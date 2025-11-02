#!/usr/bin/env python3
"""
Script simple para descargar solo Food-11
"""

import os
import sys

def main():
    print("="*70)
    print("📦 DESCARGA DE FOOD-11 DATASET")
    print("="*70)
    
    # Verificar Kaggle
    try:
        import kaggle
        print("✓ Kaggle API encontrada\n")
    except:
        print("❌ Error: Instala Kaggle con: pip install kaggle")
        sys.exit(1)
    
    # Configuración
    destino = 'datos/originales/food11'
    dataset_id = 'trolukovich/food11-image-dataset'
    
    print(f"Dataset: {dataset_id}")
    print(f"Destino: {destino}")
    print(f"Tamaño: ~1.19 GB")
    print(f"Tiempo estimado: 3-10 minutos\n")
    
    # Verificar si ya existe
    if os.path.exists(destino) and os.listdir(destino):
        print(f"⚠️  Food-11 ya existe en {destino}")
        respuesta = input("¿Descargar de nuevo? (y/N): ").strip().lower()
        if respuesta not in ['y', 'yes', 's', 'si', 'sí']:
            print("✓ Usando Food-11 existente")
            return
    
    # Crear carpeta
    os.makedirs(destino, exist_ok=True)
    
    # Descargar
    print("⬇️  Descargando...\n")
    resultado = os.system(f'kaggle datasets download -d {dataset_id} -p {destino} --unzip')
    
    if resultado == 0:
        print("\n" + "="*70)
        print("✅ FOOD-11 DESCARGADO EXITOSAMENTE")
        print("="*70)
        
        # Verificar estructura
        print("\n📂 Estructura descargada:")
        for item in os.listdir(destino):
            item_path = os.path.join(destino, item)
            if os.path.isdir(item_path):
                num_files = sum([len(files) for r, d, files in os.walk(item_path)])
                print(f"   ├── {item}/ ({num_files} archivos)")
        
        print("\n🚀 Siguiente paso:")
        print("   Verificar datos: python scripts/verificar_food11.py")
        print("="*70)
    else:
        print("\n❌ ERROR AL DESCARGAR")
        print("Verifica:")
        print("  1. Tu conexión a internet")
        print("  2. Que kaggle.json esté en ~/.kaggle/")
        print("  3. Que tengas espacio en disco (~2 GB)")

if __name__ == '__main__':
    main()
