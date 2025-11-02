"""Verifica integridad básica de los datasets esperados.
Comprueba existencia de carpetas y cuenta imágenes.
"""
import os
from src.configuracion import Config


def contar_imagenes(dirpath):
    total = 0
    for root, _, files in os.walk(dirpath):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                total += 1
    return total


def main():
    cfg = Config()
    checks = [
        ('Nutrition5k', cfg.NUTRITION5K_DIR),
        ('Food101', cfg.FOOD101_DIR),
        ('Procesados', cfg.PROCESADOS_DIR)
    ]

    for name, path in checks:
        exists = os.path.exists(path)
        print(f"{name}: {'✓' if exists else '✗'} - {path}")
        if exists:
            print(f"  Imágenes encontradas: {contar_imagenes(path)}")

if __name__ == '__main__':
    main()
