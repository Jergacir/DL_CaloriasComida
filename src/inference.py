# ═══════════════════════════════════════════════════════════
# src/inference.py
# Sistema integrado de clasificación y estimación de calorías
# ═══════════════════════════════════════════════════════════

import torch
import sys
import os
import json
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

# Agregar rutas de modelos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modelo1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modelo2'))

from cnn_clasificador import CNNClasificador
from cnn_regressor import CNNRegressor


class SistemaCaloriasComida:
    """
    Sistema integrado de dos modelos CNN:
    - Modelo 1: Clasifica tipo de comida (Food-101: 101 categorías)
    - Modelo 2: Estima calorías (regresión)
    """
    
    def __init__(self, modelo1_path, modelo2_path=None, device='auto'):
        """
        Inicializa el sistema de inferencia
        
        Args:
            modelo1_path: Path al modelo de clasificación (.pth)
            modelo2_path: Path al modelo de regresión (.pth) [opcional]
            device: 'cuda', 'cpu' o 'auto'
        """
        # Detectar dispositivo
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔥 Dispositivo: {self.device}")
        
        # Transformaciones estándar
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Cargar Modelo 1 (Clasificador)
        self._cargar_modelo1(modelo1_path)
        
        # Cargar Modelo 2 (Regressor) si se proporciona
        if modelo2_path:
            self._cargar_modelo2(modelo2_path)
        else:
            self.modelo2 = None
            self.mean_cal = None
            self.std_cal = None
            print("⚠️  Modelo 2 no cargado (solo clasificación disponible)")
    
    def _cargar_modelo1(self, path):
        """Carga el modelo de clasificación"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo 1 no encontrado: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Cargar clases desde checkpoint
        if 'classes' in checkpoint:
            self.CLASES = {i: cls for i, cls in enumerate(checkpoint['classes'])}
            num_clases = len(self.CLASES)
        else:
            # Fallback: Food-101
            num_clases = 101
            print("⚠️  Clases no encontradas en checkpoint, asumiendo Food-101")
        
        self.modelo1 = CNNClasificador(num_clases=num_clases).to(self.device)
        self.modelo1.load_state_dict(checkpoint['model_state_dict'])
        self.modelo1.eval()
        
        print(f"✓ Modelo 1 cargado: {path} ({num_clases} clases)")
    
    def _cargar_modelo2(self, path):
        """Carga el modelo de regresión de calorías"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo 2 no encontrado: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.modelo2 = CNNRegressor().to(self.device)
        
        # Manejar diferentes formatos de guardado
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Cargar estadísticas de normalización si existen
            if 'mean_cal' in checkpoint and 'std_cal' in checkpoint:
                self.mean_cal = checkpoint['mean_cal']
                self.std_cal = checkpoint['std_cal']
                print(f"✓ Modelo 2 cargado con normalización (mean={self.mean_cal:.1f}, std={self.std_cal:.1f})")
            else:
                # ✅ VALORES DEL ENTRENAMIENTO
                self.mean_cal = 210.50
                self.std_cal = 160.22
                print(f"⚠️  Usando estadísticas del entrenamiento: mean={self.mean_cal:.1f}, std={self.std_cal:.1f}")
        else:
            state_dict = checkpoint
            # ✅ VALORES DEL ENTRENAMIENTO
            self.mean_cal = 210.50
            self.std_cal = 160.22
            print(f"⚠️  Usando estadísticas del entrenamiento: mean={self.mean_cal:.1f}, std={self.std_cal:.1f}")
        
        self.modelo2.load_state_dict(state_dict)
        self.modelo2.eval()
        
        print(f"✓ Modelo 2 cargado: {path}")
    
    def predecir(self, imagen_path, verbose=True):
        """
        Predice categoría y calorías de una imagen
        
        Args:
            imagen_path: Path a la imagen
            verbose: Mostrar información del proceso
        
        Returns:
            dict con 'clase', 'clase_id', 'probabilidad', 'calorias'
        """
        # Cargar y preprocesar imagen
        try:
            imagen = Image.open(imagen_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Error al cargar imagen: {e}")
        
        img_tensor = self.transform(imagen).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Modelo 1: Clasificación
            output1 = self.modelo1(img_tensor)
            probs = torch.softmax(output1, dim=1)
            prob_max, pred_clase = torch.max(probs, 1)
            
            clase_id = pred_clase.item()
            clase_nombre = self.CLASES.get(clase_id, f"Clase_{clase_id}")
            probabilidad = prob_max.item()
            
            # Modelo 2: Calorías (si está disponible)
            calorias = None
            if self.modelo2 is not None:
                calorias_norm = self.modelo2(img_tensor).item()
                
                # Denormalizar si hay estadísticas
                if self.mean_cal is not None and self.std_cal is not None:
                    calorias = calorias_norm * self.std_cal + self.mean_cal
                else:
                    calorias = calorias_norm
        
        resultado = {
            'clase': clase_nombre,
            'clase_id': clase_id,
            'probabilidad': round(probabilidad * 100, 2),
            'calorias': round(calorias, 1) if calorias else None
        }
        
        if verbose:
            self._mostrar_resultado(resultado)
        
        return resultado
    
    def predecir_batch(self, imagenes_paths):
        """
        Predice para múltiples imágenes
        
        Args:
            imagenes_paths: Lista de paths a imágenes
        
        Returns:
            Lista de diccionarios con resultados
        """
        resultados = []
        for i, path in enumerate(imagenes_paths, 1):
            print(f"\n[{i}/{len(imagenes_paths)}] Procesando: {os.path.basename(path)}")
            try:
                resultado = self.predecir(path, verbose=False)
                resultados.append(resultado)
            except Exception as e:
                print(f"⚠️  Error: {e}")
                resultados.append(None)
        
        return resultados
    
    def _mostrar_resultado(self, resultado):
        """Muestra resultado formateado"""
        print("\n" + "="*70)
        print("🍽️  PREDICCIÓN")
        print("="*70)
        print(f"📋 Categoría:  {resultado['clase']}")
        print(f"📊 Confianza:  {resultado['probabilidad']:.1f}%")
        if resultado['calorias']:
            print(f"🔥 Calorías:   {resultado['calorias']:.0f} kcal")
        print("="*70)
    
    def top_k_predicciones(self, imagen_path, k=3):
        """
        Obtiene las top-k predicciones más probables
        
        Args:
            imagen_path: Path a la imagen
            k: Número de predicciones a retornar
        
        Returns:
            Lista de dict con clase y probabilidad
        """
        imagen = Image.open(imagen_path).convert('RGB')
        img_tensor = self.transform(imagen).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.modelo1(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            top_probs, top_indices = torch.topk(probs, k)
        
        resultados = []
        for prob, idx in zip(top_probs, top_indices):
            resultados.append({
                'clase': self.CLASES.get(idx.item(), f"Clase_{idx.item()}"),
                'probabilidad': round(prob.item() * 100, 2)
            })
        
        return resultados


# ═══════════════════════════════════════════════════════════
# Función auxiliar para uso rápido
# ═══════════════════════════════════════════════════════════

def predecir_simple(imagen_path, modelo1_path, modelo2_path=None):
    """
    Función rápida para hacer una predicción
    
    Ejemplo:
        resultado = predecir_simple('imagen.jpg', 'modelo1.pth', 'modelo2.pth')
    """
    sistema = SistemaCaloriasComida(modelo1_path, modelo2_path)
    return sistema.predecir(imagen_path)
