# ═══════════════════════════════════════════════════════════
# app/app.py
# Backend Flask para Sistema de Estimación de Calorías
# ═══════════════════════════════════════════════════════════

from flask import Flask, render_template, request, jsonify, send_from_directory
import sys
import os
from werkzeug.utils import secure_filename
import logging
from datetime import datetime

# Importar sistema de inferencia
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from inference import SistemaCaloriasComida

# ═══════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Crear carpetas necesarias
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# CARGAR MODELOS AL INICIAR
# ═══════════════════════════════════════════════════════════

MODELO1_PATH = os.path.join(os.path.dirname(__file__), '..', 'modelos', 'modelo1_mejor.pth')
MODELO2_PATH = os.path.join(os.path.dirname(__file__), '..', 'modelos', 'modelo2_mejor.pth')

logger.info("🚀 Inicializando sistema...")

try:
    sistema = SistemaCaloriasComida(
        MODELO1_PATH, 
        MODELO2_PATH if os.path.exists(MODELO2_PATH) else None
    )
    logger.info("✅ Modelos cargados exitosamente")
except Exception as e:
    logger.error(f"❌ Error al cargar modelos: {e}")
    sistema = None

# ═══════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════

def allowed_file(filename):
    """Verifica si el archivo tiene extensión permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ═══════════════════════════════════════════════════════════
# RUTAS
# ═══════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint principal de predicción"""
    
    # Validar que hay sistema cargado
    if sistema is None:
        return jsonify({
            'success': False,
            'error': 'Modelos no cargados. Verifica que existan los archivos .pth'
        }), 500
    
    # Validar que hay archivo
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No se recibió ningún archivo'
        }), 400
    
    file = request.files['file']
    
    # Validar que se seleccionó archivo
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No se seleccionó ningún archivo'
        }), 400
    
    # Validar extensión
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Formato de archivo no permitido. Usa: jpg, jpeg, png, gif'
        }), 400
    
    try:
        # Guardar archivo con timestamp único
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(file.filename)
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"📸 Procesando: {filename}")
        
        # Hacer predicción
        resultado = sistema.predecir(filepath, verbose=False)
        
        # Top-3 predicciones
        top3 = sistema.top_k_predicciones(filepath, k=3)
        
        logger.info(f"✅ Predicción: {resultado['clase']} ({resultado['probabilidad']:.1f}%)")
        
        # Retornar resultado (mantener imagen para mostrar)
        return jsonify({
            'success': True,
            'resultado': resultado,
            'top3': top3,
            'imagen_url': f'/static/uploads/{filename}'
        })
    
    except Exception as e:
        logger.error(f"❌ Error en predicción: {e}")
        
        # Limpiar archivo si hay error
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': False,
            'error': f'Error al procesar imagen: {str(e)}'
        }), 500

@app.route('/info')
def info():
    """Información del sistema"""
    if sistema is None:
        return jsonify({
            'success': False,
            'error': 'Sistema no inicializado'
        }), 500
    
    return jsonify({
        'success': True,
        'info': {
            'clases': list(sistema.CLASES.values()),
            'num_clases': len(sistema.CLASES),
            'device': str(sistema.device),
            'modelo1_disponible': True,
            'modelo2_disponible': sistema.modelo2 is not None
        }
    })

@app.route('/health')
def health():
    """Health check para deployment"""
    return jsonify({
        'status': 'healthy',
        'sistema_cargado': sistema is not None
    })

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Servir imágenes subidas"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ═══════════════════════════════════════════════════════════
# MANEJO DE ERRORES
# ═══════════════════════════════════════════════════════════

@app.errorhandler(413)
def request_entity_too_large(error):
    """Error: archivo muy grande"""
    return jsonify({
        'success': False,
        'error': 'Archivo demasiado grande. Máximo 16 MB'
    }), 413

@app.errorhandler(404)
def not_found(error):
    """Error 404"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Error 500"""
    logger.error(f"Error 500: {error}")
    return jsonify({
        'success': False,
        'error': 'Error interno del servidor'
    }), 500

# ═══════════════════════════════════════════════════════════
# INICIAR APP
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger.info("🌐 Iniciando servidor Flask...")
    logger.info("📍 Abre tu navegador en: http://localhost:5000")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )
