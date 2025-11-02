"""Script simple para dividir un dataset de tipo ImageFolder en train/val/test
usa las proporciones definidas en src.configuracion.Config y copia imágenes a
`datos/procesados/food11/entrenamiento|validacion|prueba`.
"""
import os
import shutil
import random
from pathlib import Path
from src.configuracion import Config


def split_image_folder(source_dir, dest_root, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42):
    random.seed(seed)
    dest_root = Path(dest_root)
    for clase in os.listdir(source_dir):
        src_clase = Path(source_dir) / clase
        if not src_clase.is_dir():
            continue
        imagenes = [p for p in src_clase.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        random.shuffle(imagenes)
        n = len(imagenes)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train = imagenes[:n_train]
        val = imagenes[n_train:n_train+n_val]
        test = imagenes[n_train+n_val:]

        for subset, items in [('entrenamiento', train), ('validacion', val), ('prueba', test)]:
            dest_dir = dest_root / 'food11' / subset / clase
            dest_dir.mkdir(parents=True, exist_ok=True)
            for p in items:
                shutil.copy(p, dest_dir / p.name)

    print('División completada.')


if __name__ == '__main__':
    cfg = Config()
    source = os.path.join(cfg.FOOD101_DIR)  # ajuste si es food11
    dest = os.path.join(cfg.PROCESADOS_DIR)
    # Si el usuario tiene food11 en otra ruta, modificar `source`.
    split_image_folder(source, dest, cfg.TRAIN_SPLIT, cfg.VAL_SPLIT, cfg.TEST_SPLIT)
