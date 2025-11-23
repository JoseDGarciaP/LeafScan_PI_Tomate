"""
streamlit_app_v2.py - CORREGIDO: Sin normalización
"""
import streamlit as st
import numpy as np
from PIL import Image
import json
import time
import tensorflow as tf

st.set_page_config(
    page_title="Detector de Bacteria en Tomate",
    page_icon="🍅",
    layout="wide"
)

MODEL_PATH = 'models/model_tomato.keras'
CLASS_NAMES_PATH = 'models/class_names.json'

@st.cache_resource
def load_model():
    """Carga el modelo y las clases"""
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        
        # Cargar modelo .keras (Keras 3 compatible)
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        st.success(f"✓ Modelo cargado correctamente")
        st.info(f"TensorFlow: {tf.__version__} | Keras: {tf.keras.__version__}")
        
        return model, class_names
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        st.info("Ejecuta primero: python entrenar_desde_CERO_v2.py")
        return None, None

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocesa la imagen SIN NORMALIZACIÓN
    El modelo fue entrenado con valores [0, 255]
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32)
    # ¡SIN NORMALIZACIÓN! El modelo espera [0, 255]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, class_names, image):
    """Realiza la predicción"""
    processed_img = preprocess_image(image)
    
    start_time = time.time()
    predictions = model.predict(processed_img, verbose=0)
    inference_time = time.time() - start_time
    
    predicted_class_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = class_names[predicted_class_idx]
    
    # Debug info (opcional, puedes comentar esto después)
    print("\n" + "="*50)
    print("🔍 DEBUG - PREDICCIÓN")
    print("="*50)
    for i, cls in enumerate(class_names):
        prob = predictions[0][i] * 100
        marker = " ← PREDICHO" if i == predicted_class_idx else ""
        print(f"[{i}] {cls}: {prob:.2f}%{marker}")
    print("="*50 + "\n")
    
    return predicted_class, confidence, predictions[0], inference_time

def get_recommendation(class_name):
    """Retorna recomendaciones según la clase detectada"""
    if "healthy" in class_name.lower():
        return {
            "status": "✅ Hoja Sana",
            "color": "success",
            "description": "La hoja parece estar en buen estado.",
            "recommendations": [
                "✓ Continuar con cuidados regulares",
                "✓ Mantener monitoreo periódico",
                "✓ Asegurar riego adecuado",
                "✓ Revisar semanalmente"
            ]
        }
    else:
        return {
            "status": "⚠️ Bacteria Detectada",
            "color": "error",
            "description": "Se detectó presencia de mancha bacteriana.",
            "recommendations": [
                "🔸 Aislar la planta afectada inmediatamente",
                "🔸 Remover hojas afectadas con tijeras esterilizadas",
                "🔸 Considerar tratamiento con bactericidas a base de cobre",
                "🔸 Mejorar la ventilación del cultivo",
                "🔸 Evitar el riego por aspersión",
                "🔸 Consultar con un agrónomo especializado"
            ]
        }

def main():
    # Cargar modelo
    model, class_names = load_model()
    
    if model is None:
        st.stop()
    
    # Título
    st.title("🍅 Detector de Bacteria en Hojas de Tomate")
    st.markdown("### Sistema de diagnóstico con Inteligencia Artificial")
    
    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información del Sistema")
        
        st.markdown("**Modelo:** MobileNetV2")
        st.markdown("**Arquitectura:** Transfer Learning")
        st.markdown(f"**TensorFlow:** {tf.__version__}")
        
        st.markdown("---")
        st.markdown("**Clases detectadas:**")
        for i, cls in enumerate(class_names):
            emoji = "🌿" if "healthy" in cls.lower() else "🦠"
            st.markdown(f"{emoji} **{i}.** {cls}")
        
        st.markdown("---")
        st.markdown("**Instrucciones:**")
        st.markdown("""
        1. Carga una imagen de hoja de tomate
        2. Presiona el botón 'Analizar'
        3. Revisa los resultados y recomendaciones
        """)
        
        st.markdown("---")
        st.markdown("**Consejos para mejores resultados:**")
        st.markdown("""
        - Usa buena iluminación
        - Enfoca bien la hoja
        - Evita sombras pronunciadas
        - Imagen clara y nítida
        """)
    
    # Layout principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Cargar Imagen")
        
        uploaded_file = st.file_uploader(
            "Selecciona una imagen de hoja de tomate",
            type=['jpg', 'jpeg', 'png'],
            help="Formatos aceptados: JPG, JPEG, PNG"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", use_container_width=True)
            
            if st.button("🔍 Analizar Hoja", type="primary", use_container_width=True):
                with st.spinner("Analizando imagen..."):
                    pred_class, conf, all_pred, inf_time = predict(
                        model, class_names, image
                    )
                    rec = get_recommendation(pred_class)
                    
                    # Guardar en session_state
                    st.session_state.update({
                        'prediction': pred_class,
                        'confidence': conf,
                        'all_predictions': all_pred,
                        'inference_time': inf_time,
                        'recommendation': rec
                    })
                    
                    st.rerun()
    
    with col2:
        st.subheader("📊 Resultados del Análisis")
        
        if 'prediction' in st.session_state:
            rec = st.session_state['recommendation']
            conf = st.session_state['confidence']
            all_pred = st.session_state['all_predictions']
            inf_time = st.session_state['inference_time']
            
            # Mostrar resultado principal
            if rec['color'] == 'success':
                st.success(f"## {rec['status']}")
            else:
                st.error(f"## {rec['status']}")
            
            # Métricas
            col_conf, col_time = st.columns(2)
            with col_conf:
                st.metric("Confianza", f"{conf*100:.1f}%")
            with col_time:
                st.metric("Tiempo", f"{inf_time*1000:.0f} ms")
            
            # Barra de progreso
            st.progress(conf)
            
            # Diagnóstico
            st.markdown("---")
            st.markdown(f"**Diagnóstico:** {rec['description']}")
            
            # Recomendaciones
            st.markdown("**Recomendaciones:**")
            for item in rec['recommendations']:
                st.markdown(f"{item}")
            
            # Detalles técnicos (expandible)
            with st.expander("🔧 Detalles Técnicos"):
                st.write(f"**Clase predicha:** {st.session_state['prediction']}")
                st.write(f"**Índice:** {np.argmax(all_pred)}")
                st.write(f"**Tiempo de inferencia:** {inf_time*1000:.2f} ms")
                
                st.markdown("**Probabilidades por clase:**")
                for i, cls in enumerate(class_names):
                    prob = all_pred[i] * 100
                    st.write(f"- **{cls}:** {prob:.2f}%")
            
            # Botón para nueva predicción
            if st.button("🔄 Analizar otra imagen", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        else:
            st.info("👆 Carga una imagen y presiona 'Analizar Hoja' para comenzar")
            
            # Imagen de ejemplo
            st.markdown("---")
            st.markdown("**Ejemplo de imagen ideal:**")
            st.markdown("""
            - Hoja completa visible
            - Buena iluminación
            - Enfoque claro
            - Fondo uniforme (opcional)
            """)

if __name__ == '__main__':
    main()