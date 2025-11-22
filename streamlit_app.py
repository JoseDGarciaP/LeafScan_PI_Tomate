"""
streamlit_app.py - Aplicación web principal
"""
import streamlit as st
import numpy as np
from PIL import Image
import json
import time

try:
    import tflite_runtime.interpreter as tflite
    USING_TFLITE_RUNTIME = True
except ImportError:
    import tensorflow as tf
    USING_TFLITE_RUNTIME = False

st.set_page_config(
    page_title="Detector de Bacteria en Tomate",
    page_icon="🍅",
    layout="wide"
)

MODEL_PATH = 'models/model_tomato_quantized.tflite'
CLASS_NAMES_PATH = 'models/class_names.json'

@st.cache_resource
def load_model():
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        
        if USING_TFLITE_RUNTIME:
            interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        else:
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        return interpreter, input_details, output_details, class_names
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Entrena el modelo primero con: python src/train.py")
        return None, None, None, None

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(interpreter, input_details, output_details, class_names, image):
    processed_img = preprocess_image(image)
    
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], processed_img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    inference_time = time.time() - start_time
    
    predicted_class_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, predictions[0], inference_time

def get_recommendation(class_name):
    if "healthy" in class_name.lower():
        return {
            "status": "✅ Hoja Sana",
            "description": "La hoja parece estar en buen estado.",
            "recommendations": [
                "Continuar con cuidados regulares",
                "Mantener monitoreo periódico",
                "Asegurar riego adecuado"
            ]
        }
    else:
        return {
            "status": "⚠️ Bacteria Detectada",
            "description": "Se detectó presencia de bacteria.",
            "recommendations": [
                "Aislar la planta afectada",
                "Considerar tratamiento con bactericidas",
                "Mejorar ventilación",
                "Consultar con agrónomo",
                "Remover hojas afectadas"
            ]
        }

def main():
    interpreter, input_details, output_details, class_names = load_model()
    
    if interpreter is None:
        st.stop()
    
    st.title("🍅 Detector de Bacteria en Hojas de Tomate")
    st.markdown("### Sistema de diagnóstico con IA")
    
    with st.sidebar:
        st.header("ℹ️ Información")
        st.markdown("""
        **Modelo:** MobileNetV2
        **Clases:** Sana / Bacteria
        """)
        runtime = "TFLite Runtime" if USING_TFLITE_RUNTIME else "TensorFlow"
        st.info(f"**Runtime:** {runtime}")
        
        st.markdown("**Clases:**")
        for i, cls in enumerate(class_names):
            st.write(f"{i+1}. {cls}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Cargar Imagen")
        uploaded_file = st.file_uploader(
            "Selecciona una imagen",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", use_container_width=True)
            
            if st.button("🔍 Analizar", type="primary", use_container_width=True):
                with st.spinner("Analizando..."):
                    pred_class, conf, all_pred, inf_time = predict(
                        interpreter, input_details, output_details, class_names, image
                    )
                    rec = get_recommendation(pred_class)
                    
                    st.session_state.update({
                        'prediction': pred_class,
                        'confidence': conf,
                        'all_predictions': all_pred,
                        'inference_time': inf_time,
                        'recommendation': rec
                    })
    
    with col2:
        st.subheader("📊 Resultados")
        
        if 'prediction' in st.session_state:
            rec = st.session_state['recommendation']
            conf = st.session_state['confidence']
            
            st.markdown(f"### {rec['status']}")
            st.metric("Confianza", f"{conf*100:.2f}%")
            st.progress(conf)
            
            st.markdown(f"**Diagnóstico:** {rec['description']}")
            st.markdown("**Recomendaciones:**")
            for item in rec['recommendations']:
                st.markdown(f"- {item}")
            
            with st.expander("🔧 Detalles Técnicos"):
                st.write(f"**Clase:** {st.session_state['prediction']}")
                st.write(f"**Tiempo:** {st.session_state['inference_time']*1000:.2f} ms")
                
                st.markdown("**Probabilidades:**")
                for i, cls in enumerate(class_names):
                    prob = st.session_state['all_predictions'][i] * 100
                    st.write(f"- {cls}: {prob:.2f}%")
        else:
            st.info("👆 Carga una imagen y presiona Analizar")

if __name__ == '__main__':
    main()