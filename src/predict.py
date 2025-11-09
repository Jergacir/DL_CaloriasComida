#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════
# src/predict.py
# Script de línea de comandos para predicción
# ═══════════════════════════════════════════════════════════

import argparse
import os
from inference import SistemaCaloriasComida


def main():
    parser = argparse.ArgumentParser(
        description='Sistema de Estimación de Calorías con Deep Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Predicción básica (solo clasificación)
  python predict.py --imagen foto.jpg --modelo1 modelos/modelo1_mejor.pth
  
  # Predicción completa (clasificación + calorías)
  python predict.py --imagen foto.jpg \\
                    --modelo1 modelos/modelo1_mejor.pth \\
                    --modelo2 modelos/modelo2_mejor.pth
  
  # Top-3 predicciones
  python predict.py --imagen foto.jpg --modelo1 modelos/modelo1_mejor.pth --top 3
        """
    )
    
    parser.add_argument('--imagen', type=str, required=True,
                        help='Path a la imagen de comida')
    parser.add_argument('--modelo1', type=str, required=True,
                        help='Path al modelo de clasificación (.pth)')
    parser.add_argument('--modelo2', type=str, default=None,
                        help='Path al modelo de regresión (.pth) [opcional]')
    parser.add_argument('--top', type=int, default=1,
                        help='Mostrar top-k predicciones (default: 1)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Dispositivo de cómputo (default: auto)')
    
    args = parser.parse_args()
    
    # Validar archivo de imagen
    if not os.path.exists(args.imagen):
        print(f"❌ Error: Imagen no encontrada: {args.imagen}")
        return
    
    # Cargar sistema
    print("🚀 Inicializando sistema...")
    try:
        sistema = SistemaCaloriasComida(args.modelo1, args.modelo2, device=args.device)
    except Exception as e:
        print(f"❌ Error al cargar modelos: {e}")
        return
    
    # Predecir
    print(f"\n📸 Procesando imagen: {args.imagen}")
    
    if args.top > 1:
        # Top-k predicciones
        resultados = sistema.top_k_predicciones(args.imagen, k=args.top)
        print("\n" + "="*70)
        print(f"🏆 TOP-{args.top} PREDICCIONES")
        print("="*70)
        for i, res in enumerate(resultados, 1):
            print(f"{i}. {res['clase']:<20} {res['probabilidad']:>6.2f}%")
        print("="*70)
    else:
        # Predicción simple
        resultado = sistema.predecir(args.imagen)
    
    print("\n✅ Predicción completada")


if __name__ == '__main__':
    main()
