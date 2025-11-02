#!/usr/bin/env python3
"""
Verifica que Food-11 se descargó correctamente
"""

import os

def verificar_food11():
    print("="*70)
    print("🔍 VERIFICACIÓN DE FOOD-11")
    print("="*70)
    
    root = 'datos/originales/food11'
    
    # Verificar que existe
    if not os.path.exists(root):
        print(f"❌ Error: {root} no existe")
        print("Ejecuta primero: python scripts/descargar_food11.py")
        return False
    
    # Splits esperados
    splits = ['training', 'validation', 'evaluation']
    
    # Clases esperadas
    clases = [
        'Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food',
        'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 
        'Vegetable-Fruit'
    ]
    
    print(f"\n📂 Directorio: {root}")
    
    # Verificar cada split
    total_imagenes = 0
    todo_ok = True
    
    for split in splits:
        split_path = os.path.join(root, split)
        
        if not os.path.exists(split_path):
            print(f"\n❌ {split}/ NO ENCONTRADO")
            todo_ok = False
            continue
        
        print(f"\n✓ {split}/")
        
        # Contar imágenes por clase
        for clase in clases:
            clase_path = os.path.join(split_path, clase)
            
            if os.path.exists(clase_path):
                imagenes = [f for f in os.listdir(clase_path) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))]
                num_imgs = len(imagenes)
                total_imagenes += num_imgs
                print(f"   ├── {clase}: {num_imgs} imágenes")
            else:
                print(f"   ├── {clase}: ❌ NO ENCONTRADO")
                todo_ok = False
    
    # Resumen
    print("\n" + "="*70)
    if todo_ok:
        print(f"✅ FOOD-11 VERIFICADO CORRECTAMENTE")
        print(f"\n📊 Resumen:")
        print(f"   - Total de imágenes: {total_imagenes}")
        print(f"   - Splits: {len(splits)}")
        print(f"   - Clases: {len(clases)}")
        print(f"\n🚀 Listo para usar en entrenamiento")
    else:
        print("❌ PROBLEMAS ENCONTRADOS")
        print("Intenta descargar de nuevo")
    print("="*70)
    
    return todo_ok

if __name__ == '__main__':
    verificar_food11()
