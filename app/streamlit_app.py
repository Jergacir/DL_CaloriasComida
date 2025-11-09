# ═══════════════════════════════════════════════════════════
# app/streamlit_app.py
# Web app con Streamlit para clasificación y estimación de calorías
# ═══════════════════════════════════════════════════════════

import streamlit as st
import sys
import os
from PIL import Image
import io

# Agregar path para importar inference
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from inference import SistemaCaloriasComida

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Estimación de Calorías",
    page_icon="🍽️",
    layout="wide"
)

# ═══════════════════════════════════════════════════════════
# TÍTULO Y DESCRIPCIÓN
# ═══════════════════════════════════════════════════════════

st.title("🍽️ Sistema de Estimación de Calorías con Deep Learning")
st.markdown("""
### 📊 ¿Qué hace esta aplicación?
- **Clasifica** el tipo de comida en 11 categorías
- **Estima** las calorías automáticamente
- Usa **dos modelos CNN** entrenados en Food-11 y Nutrition5k
""")

st.divider()

# ═══════════════════════════════════════════════════════════
# SIDEBAR: Configuración
# ═══════════════════════════════════════════════════════════

st.sidebar.header("⚙️ Configuración")

# Paths de modelos
MODELO1_PATH = st.sidebar.text_input(
    "Path Modelo 1 (Clasificador)",
    value="../modelos/modelo1_mejor.pth"
)

MODELO2_PATH = st.sidebar.text_input(
    "Path Modelo 2 (Calorías)",
    value="../modelos/modelo2_mejor.pth"
)

# Botón para cargar modelos
if st.sidebar.button("🚀 Cargar Modelos"):
    with st.spinner("Cargando modelos..."):
        try:
            st.session_state['sistema'] = SistemaCaloriasComida(
                MODELO1_PATH, 
                MODELO2_PATH if MODELO2_PATH else None
            )
            st.sidebar.success("✅ Modelos cargados")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")

st.sidebar.divider()

# Información
st.sidebar.header("📖 Información")
st.sidebar.markdown("""
**Categorías disponibles:**
- Bread
- Dairy product
- Dessert
- Egg
- Fried food
- Meat
- Noodles/Pasta
- Rice
- Seafood
- Soup
- Vegetable/Fruit

**Rendimiento:**
- Clasificador: 58.29% accuracy
- Regresión: 46.15 kcal MAE
""")

# ═══════════════════════════════════════════════════════════
# ÁREA PRINCIPAL: Upload y Predicción
# ═══════════════════════════════════════════════════════════

st.header("📸 Subir Imagen")

# File uploader
uploaded_file = st.file_uploader(
    "Arrastra una imagen o haz clic para seleccionar",
    type=['jpg', 'jpeg', 'png'],
    help="Sube una imagen de comida para clasificar y estimar calorías"
)

if uploaded_file is not None:
    # Mostrar imagen
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Imagen Original")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("📊 Predicción")
        
        # Verificar que el sistema esté cargado
        if 'sistema' not in st.session_state:
            st.warning("⚠️ Primero carga los modelos desde el sidebar")
        else:
            # Botón para predecir
            if st.button("🔮 Predecir", type="primary"):
                with st.spinner("Analizando imagen..."):
                    try:
                        # Guardar imagen temporal
                        temp_path = "temp_image.jpg"
                        image.save(temp_path)
                        
                        # Hacer predicción
                        sistema = st.session_state['sistema']
                        resultado = sistema.predecir(temp_path, verbose=False)
                        
                        # Mostrar resultados
                        st.success("✅ Predicción completada")
                        
                        # Métricas
                        met1, met2, met3 = st.columns(3)
                        with met1:
                            st.metric("Categoría", resultado['clase'])
                        with met2:
                            st.metric("Confianza", f"{resultado['probabilidad']:.1f}%")
                        with met3:
                            if resultado['calorias']:
                                st.metric("Calorías", f"{resultado['calorias']:.0f} kcal")
                            else:
                                st.metric("Calorías", "N/A")
                        
                        # Top-3 predicciones
                        st.divider()
                        st.subheader("🏆 Top-3 Predicciones")
                        
                        top3 = sistema.top_k_predicciones(temp_path, k=3)
                        for i, pred in enumerate(top3, 1):
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.write(f"{i}. **{pred['clase']}**")
                            with col_b:
                                st.write(f"{pred['probabilidad']:.1f}%")
                        
                        # Limpiar archivo temporal
                        os.remove(temp_path)
                        
                    except Exception as e:
                        st.error(f"❌ Error en predicción: {e}")

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════

st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>🧠 Desarrollado con PyTorch y Streamlit</p>
    <p>📚 Datasets: Food-11 y Nutrition5k</p>
</div>
""", unsafe_allow_html=True)
