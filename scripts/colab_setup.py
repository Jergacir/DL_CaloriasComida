# scripts/colab_setup.py

"""
Script de setup automático para Google Colab.
Ejecutar al inicio de cada sesión de Colab.

Uso en Colab:
    !python scripts/colab_setup.py
"""

import os
import sys

def setup_colab():
    """Configura el entorno de Colab automáticamente"""
    
    print("="*70)
    print("🔧 CONFIGURACIÓN AUTOMÁTICA DE GOOGLE COLAB")
    print("="*70)
    
    # 1. Verificar GPU
    print("\n[1/6] Verificando GPU...")
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("   ⚠️  GPU no disponible. Activar en Runtime → Change runtime type")
    
    # 2. Montar Drive
    print("\n[2/6] Montando Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print("   ✓ Drive montado")
    except:
        print("   ⚠️  No se pudo montar Drive (ejecuta manualmente)")
    
    # 3. Clonar repo (si no existe)
    print("\n[3/6] Clonando/Actualizando repositorio...")
    repo_url = 'https://github.com/Jergacir/DL_CaloriasComida.git'  # ← CAMBIAR por tu URL
    
    if not os.path.exists('/content/DL_CaloriasComida'):
        print(f"   Clonando desde {repo_url}...")
        resultado = os.system(f'git clone {repo_url}')
        if resultado == 0:
            print("   ✓ Repositorio clonado")
        else:
            print("   ❌ Error clonando repositorio")
            return False
    else:
        print("   ✓ Repositorio ya existe, actualizando...")
        os.chdir('/content/DL_CaloriasComida')
        os.system('git pull')
    
    os.chdir('/content/DL_CaloriasComida')
    print(f"   ✓ Directorio actual: {os.getcwd()}")
    
    # 4. Instalar dependencias
    print("\n[4/6] Instalando dependencias...")
    resultado = os.system('pip install -q -r requirements.txt')
    if resultado == 0:
        print("   ✓ Dependencias instaladas")
    else:
        print("   ⚠️  Error instalando dependencias")
    
    # 5. Configurar datos
    print("\n[5/6] Configurando acceso a datos...")
    
    drive_food11 = '/content/drive/MyDrive/DL_CaloriasComida/datos/originales/food11'
    colab_food11 = '/content/DL_CaloriasComida/datos/originales/food11'
    
    os.makedirs('datos/originales', exist_ok=True)
    
    if not os.path.exists(colab_food11):
        if os.path.exists(drive_food11):
            try:
                os.symlink(drive_food11, colab_food11)
                print("   ✓ Food-11 enlazado desde Drive")
            except:
                print("   ⚠️  No se pudo crear enlace simbólico")
        else:
            print(f"   ⚠️  Food-11 no encontrado en Drive")
            print(f"   Esperado en: {drive_food11}")
    else:
        print("   ✓ Food-11 ya configurado")
    
    # 6. Verificar
    print("\n[6/6] Verificando configuración...")
    
    # Verificar estructura
    archivos_importantes = [
        'src/modelo1/train_m1.py',
        'src/modelo1/dataset.py',
        'src/modelo1/cnn_clasificador.py',
        'requirements.txt'
    ]
    
    todos_ok = True
    for archivo in archivos_importantes:
        if os.path.exists(archivo):
            print(f"   ✓ {archivo}")
        else:
            print(f"   ❌ {archivo} NO ENCONTRADO")
            todos_ok = False
    
    # Contar imágenes
    if os.path.exists('datos/originales/food11/training'):
        try:
            bread_path = 'datos/originales/food11/training/Bread'
            if os.path.exists(bread_path):
                num_imgs = len([f for f in os.listdir(bread_path) 
                               if f.endswith('.jpg')])
                print(f"   ✓ Datos accesibles (ej: {num_imgs} imágenes en Bread)")
            else:
                print("   ⚠️  Carpeta Bread no encontrada")
        except:
            print("   ⚠️  Error verificando datos")
    else:
        print("   ❌ Datos no accesibles")
    
    print("\n" + "="*70)
    if todos_ok:
        print("✅ CONFIGURACIÓN COMPLETADA")
    else:
        print("⚠️  CONFIGURACIÓN COMPLETADA CON ADVERTENCIAS")
    print("="*70)
    print("\n🚀 Próximos pasos:")
    print("   1. Entrenar: !python src/modelo1/train_m1.py")
    print("   2. Ver estructura: !tree -L 2")
    print("   3. Verificar GPU: !nvidia-smi")
    
    return todos_ok

if __name__ == '__main__':
    setup_colab()
