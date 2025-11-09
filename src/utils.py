# ═══════════════════════════════════════════════════════════
# src/utils.py
# Utilidades para inferencia
# ═══════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def visualizar_prediccion(imagen_path, resultado):
    """
    Visualiza imagen con predicción
    
    Args:
        imagen_path: Path a la imagen
        resultado: Dict con resultado de predicción
    """
    # Cargar imagen
    img = Image.open(imagen_path)
    
    # Crear figura
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    
    # Título con predicción
    titulo = f"{resultado['clase']}\n"
    titulo += f"Confianza: {resultado['probabilidad']:.1f}%"
    if resultado['calorias']:
        titulo += f"\nCalorías: {resultado['calorias']:.0f} kcal"
    
    plt.title(titulo, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def comparar_predicciones(imagenes_paths, sistema):
    """
    Compara predicciones de múltiples imágenes
    
    Args:
        imagenes_paths: Lista de paths a imágenes
        sistema: Instancia de SistemaCaloriasComida
    """
    n = len(imagenes_paths)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, img_path in enumerate(imagenes_paths):
        # Predicción
        resultado = sistema.predecir(img_path, verbose=False)
        
        # Mostrar imagen
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # Título
        titulo = f"{resultado['clase']}\n{resultado['probabilidad']:.1f}%"
        if resultado['calorias']:
            titulo += f"\n{resultado['calorias']:.0f} kcal"
        axes[i].set_title(titulo, fontsize=12, fontweight='bold')
    
    # Ocultar axes extras
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def exportar_resultados_csv(imagenes_paths, resultados, output_file='resultados.csv'):
    """
    Exporta resultados a CSV
    
    Args:
        imagenes_paths: Lista de paths
        resultados: Lista de resultados
        output_file: Archivo de salida
    """
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Imagen', 'Clase', 'Probabilidad (%)', 'Calorías (kcal)'])
        
        for img_path, res in zip(imagenes_paths, resultados):
            if res:
                writer.writerow([
                    img_path,
                    res['clase'],
                    res['probabilidad'],
                    res['calorias'] if res['calorias'] else 'N/A'
                ])
    
    print(f"✓ Resultados exportados a: {output_file}")
