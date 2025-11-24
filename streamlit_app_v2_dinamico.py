"""
streamlit_app_v2_dinamico.py - Con recomendaciones DINÁMICAS
Las recomendaciones cambian según el análisis de la imagen
"""
import streamlit as st
import numpy as np
from PIL import Image
import json
import time
import tensorflow as tf
import cv2

st.set_page_config(
    page_title="LeafScan Pi",
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
        
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        st.success(f"✓ Modelo cargado correctamente")
        st.info(f"TensorFlow: {tf.__version__} | Keras: {tf.keras.__version__}")
        
        return model, class_names
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        st.info("Ejecuta primero: python entrenar_desde_CERO_v2.py")
        return None, None

def analyze_image_features(image):
    """
    Analiza características visuales de la imagen para dar contexto
    Retorna: severidad estimada y características
    """
    # Convertir a array numpy
    img_array = np.array(image)
    
    # Convertir a HSV para análisis de color
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Analizar tonos marrones/amarillos (indicadores de enfermedad)
    # Hue: marrón/amarillo está en 10-30
    brown_mask = cv2.inRange(img_hsv, np.array([10, 50, 50]), np.array([30, 255, 255]))
    brown_percentage = (np.sum(brown_mask > 0) / brown_mask.size) * 100
    
    # Analizar tonos verdes oscuros (manchas)
    dark_green_mask = cv2.inRange(img_hsv, np.array([35, 40, 20]), np.array([85, 255, 120]))
    dark_spots_percentage = (np.sum(dark_green_mask > 0) / dark_green_mask.size) * 100
    
    # Calcular varianza de color (textura irregular = posible enfermedad)
    color_variance = np.std(img_array)
    
    # Estimar severidad (0-100)
    severity = min(100, brown_percentage * 2 + dark_spots_percentage * 1.5 + color_variance * 0.3)
    
    features = {
        'brown_percentage': brown_percentage,
        'dark_spots_percentage': dark_spots_percentage,
        'color_variance': color_variance,
        'severity': severity
    }
    
    return features

def preprocess_image(image, target_size=(224, 224)):
    """Preprocesa la imagen SIN NORMALIZACIÓN"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32)
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
    
    return predicted_class, confidence, predictions[0], inference_time

def get_dynamic_recommendation(class_name, confidence, features):
    """
    Genera recomendaciones DINÁMICAS basadas en:
    - Clase predicha
    - Nivel de confianza
    - Características de la imagen (severidad)
    """
    severity = features['severity']
    
    if "healthy" in class_name.lower():
        # HOJA SANA
        if confidence > 0.95:
            # Muy confiado que está sana
            return {
                "status": "✅ Hoja Completamente Sana",
                "color": "success",
                "confidence_level": "muy_alta",
                "description": "La hoja está en excelente estado. No se detectaron signos de enfermedad.",
                "recommendations": [
                    "✓ Continuar con el plan de cuidados actual",
                    "✓ Mantener monitoreo semanal preventivo",
                    "✓ Registrar esta planta como referencia de salud",
                    "✓ Asegurar condiciones ambientales óptimas"
                ],
                "preventive_tips": [
                    "💧 Riego: Mantener humedad constante sin encharcamiento",
                    "☀️ Luz: 6-8 horas de sol directo diario",
                    "🌡️ Temperatura: 21-27°C óptimo"
                ]
            }
        elif confidence > 0.80:
            # Moderadamente confiado
            return {
                "status": "✅ Hoja Aparentemente Sana",
                "color": "success",
                "confidence_level": "alta",
                "description": "La hoja parece saludable, aunque se recomienda monitoreo.",
                "recommendations": [
                    "✓ Realizar inspección visual detallada",
                    "✓ Revisar el envés de la hoja cuidadosamente",
                    "✓ Monitorear en los próximos 3-5 días",
                    "✓ Comparar con otras hojas de la planta"
                ],
                "preventive_tips": [
                    "⚠️ Verificar que no haya síntomas iniciales",
                    "🔍 Buscar pequeñas manchas o decoloración",
                    "📸 Tomar foto de seguimiento"
                ]
            }
        else:
            # Baja confianza - dudoso
            return {
                "status": "⚠️ Revisión Necesaria",
                "color": "warning",
                "confidence_level": "media",
                "description": "El análisis no es concluyente. Se recomienda inspección manual.",
                "recommendations": [
                    "🔍 Inspeccionar la hoja manualmente con lupa",
                    "📸 Tomar nuevas fotos con mejor iluminación",
                    "👨‍🌾 Consultar con un experto si hay dudas",
                    "⏰ Re-analizar en 24-48 horas"
                ],
                "preventive_tips": [
                    "💡 Mejora la iluminación para nueva foto",
                    "📐 Asegura que la hoja esté bien enfocada",
                    "🌿 Limpia la hoja suavemente antes de fotografiar"
                ]
            }
    
    else:
        # BACTERIA DETECTADA
        if severity > 60 or confidence > 0.90:
            # Infección severa o alta confianza
            return {
                "status": "🚨 Bacteria Detectada - Acción Urgente",
                "color": "error",
                "confidence_level": "muy_alta",
                "severity_level": "alta" if severity > 60 else "moderada",
                "description": f"Se detectó mancha bacteriana con {confidence*100:.1f}% de confianza. La severidad estimada es {'ALTA' if severity > 60 else 'moderada'}.",
                "recommendations": [
                    "🚨 URGENTE: Aislar la planta inmediatamente",
                    "✂️ Remover todas las hojas afectadas con tijeras esterilizadas",
                    "🧪 Aplicar bactericida de cobre dentro de las 24 horas",
                    "🔥 Destruir (quemar) hojas infectadas - NO compostar",
                    "🧼 Desinfectar herramientas después del uso",
                    "📍 Marcar plantas circundantes para monitoreo intensivo"
                ],
                "treatment_plan": [
                    "Día 1: Poda de hojas afectadas + Primera aplicación de bactericida",
                    "Día 3: Segunda aplicación de bactericida",
                    "Día 7: Tercera aplicación + Evaluación de progreso",
                    "Día 14: Inspección final y decisión de continuidad"
                ],
                "preventive_tips": [
                    "⚠️ No regar por aspersión - solo riego por goteo",
                    "🌬️ Mejorar ventilación entre plantas",
                    "🦠 Evitar trabajar con plantas cuando estén mojadas",
                    "📊 Documentar evolución con fotos diarias"
                ]
            }
        
        elif severity > 30 or confidence > 0.75:
            # Infección moderada
            return {
                "status": "⚠️ Bacteria Detectada - Tratamiento Necesario",
                "color": "warning",
                "confidence_level": "alta",
                "severity_level": "moderada",
                "description": f"Se detectó presencia bacteriana. Severidad estimada: MODERADA ({severity:.1f}%).",
                "recommendations": [
                    "⚡ Aislar la planta en las próximas 12 horas",
                    "✂️ Remover hojas visiblemente afectadas",
                    "🧪 Preparar tratamiento con bactericida cúprico",
                    "🔍 Inspeccionar plantas en un radio de 2 metros",
                    "📋 Iniciar registro de tratamiento y seguimiento"
                ],
                "treatment_plan": [
                    "Día 1-2: Poda sanitaria + Primera aplicación",
                    "Día 5: Segunda aplicación preventiva",
                    "Día 10: Evaluación y tercera aplicación si es necesario",
                    "Día 15: Revisión y ajuste del plan"
                ],
                "preventive_tips": [
                    "💧 Reducir humedad ambiental si es posible",
                    "🌡️ Monitorear temperatura (> 27°C favorece bacteria)",
                    "✋ Limitar manipulación de plantas hasta tratamiento",
                    "📸 Fotografiar evolución cada 3 días"
                ]
            }
        
        else:
            # Infección temprana o baja confianza
            return {
                "status": "⚠️ Posible Bacteria - Monitoreo Cercano",
                "color": "warning",
                "confidence_level": "media",
                "severity_level": "baja",
                "description": "Se detectaron indicios de bacteria. Requiere confirmación y monitoreo.",
                "recommendations": [
                    "🔍 Inspección manual detallada inmediata",
                    "📸 Tomar fotos adicionales de diferentes ángulos",
                    "⏰ Re-analizar en 24 horas",
                    "✂️ Preparar para poda preventiva si empeora",
                    "👨‍🌾 Considerar consulta con agrónomo"
                ],
                "treatment_plan": [
                    "Día 1: Observación + Fotos de referencia",
                    "Día 2: Re-análisis con el sistema",
                    "Día 3: Decisión de tratamiento basada en evolución",
                    "Continuar monitoreo diario por 7 días"
                ],
                "preventive_tips": [
                    "🌿 Mejorar ventilación general del cultivo",
                    "💧 Revisar prácticas de riego",
                    "🧼 Aumentar higiene de herramientas",
                    "📊 Comparar con hojas de plantas vecinas",
                    "💡 Capturar nueva imagen con mejor calidad"
                ]
            }

def main():
    model, class_names = load_model()
    
    if model is None:
        st.stop()
    
    st.title("🍅 LeafScan Pi: Detector de Salud de Cultivos de Tomate")
    st.markdown("### Sistema de diagnóstico con recomendaciones personalizadas")
    
    with st.sidebar:
        st.header("Información del Sistema")
        
        st.markdown("**Modelo:** MobileNetV2")
        st.markdown("**Características:**")
        st.markdown("- Análisis de confianza adaptativo")
        st.markdown("- Estimación de severidad")
        st.markdown("- Recomendaciones dinámicas")
        
        st.markdown("---")
        st.markdown("**Clases detectadas:**")
        for i, cls in enumerate(class_names):
            emoji = "🌿" if "healthy" in cls.lower() else "🦠"
            st.markdown(f"{emoji} **{i}.** {cls}")
        
        st.markdown("---")
        st.info("💡 **Tip:** Las recomendaciones cambian según la severidad y confianza del análisis.")
        
        # AÑADIR: Guía rápida en sidebar
        st.markdown("---")
        with st.expander("📸 **Guía para Mejores Fotos**", expanded=False):
            st.success("**✅ SÍ hacer:**")
            st.markdown("""
            - Luz natural difusa
            - Imagen nítida
            - Hoja centrada
            - Fondo uniforme
            """)
            
            st.error("**❌ NO hacer:**")
            st.markdown("""
            - Fotos borrosas
            - Sombras fuertes
            - Hoja muy pequeña
            - Múltiples hojas
            """)
            
            st.info("**🕐 Mejor momento:**")
            st.markdown("""
            - Mañana: 8-10 AM
            - Tarde: 4-6 PM
            - Evitar: mediodía
            """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Cargar Imagen")
        
        uploaded_file = st.file_uploader(
            "Selecciona una imagen de hoja de tomate",
            type=['jpg', 'jpeg', 'png'],
            help="Mejores resultados con: buena luz, hoja enfocada, sin sombras"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", width='stretch')
            
            if st.button("🔍 Analizar Hoja", type="primary", use_container_width=True):
                with st.spinner("Analizando imagen y generando recomendaciones..."):
                    # Analizar características visuales
                    features = analyze_image_features(image)
                    
                    # Predicción con el modelo
                    pred_class, conf, all_pred, inf_time = predict(
                        model, class_names, image
                    )
                    
                    # Generar recomendaciones dinámicas
                    rec = get_dynamic_recommendation(pred_class, conf, features)
                    
                    st.session_state.update({
                        'prediction': pred_class,
                        'confidence': conf,
                        'all_predictions': all_pred,
                        'inference_time': inf_time,
                        'recommendation': rec,
                        'features': features
                    })
                    
                    st.rerun()
    
    with col2:
        st.subheader("📊 Resultados del Análisis")
        
        if 'prediction' in st.session_state:
            rec = st.session_state['recommendation']
            conf = st.session_state['confidence']
            all_pred = st.session_state['all_predictions']
            features = st.session_state['features']
            inf_time = st.session_state['inference_time']  # ← AÑADIDO
            
            # Resultado principal con color dinámico
            if rec['color'] == 'success':
                st.success(f"## {rec['status']}")
            elif rec['color'] == 'warning':
                st.warning(f"## {rec['status']}")
            else:
                st.error(f"## {rec['status']}")
            
            # Métricas principales
            col_conf, col_time = st.columns(2)
            with col_conf:
                conf_color = "normal" if conf > 0.8 else "inverse"
                st.metric("Confianza", f"{conf*100:.1f}%", 
                         delta=f"{rec['confidence_level']}")
            with col_time:
                st.metric("Tiempo", f"{inf_time*1000:.0f} ms")
            
            # Barra de progreso con color
            st.progress(conf)
            
            # Diagnóstico
            st.markdown("---")
            st.markdown(f"**📋 Diagnóstico:** {rec['description']}")
            
            # Severidad (solo si es bacteria)
            if 'severity_level' in rec:
                severity_emoji = {"baja": "🟢", "moderada": "🟡", "alta": "🔴"}
                severity_label = rec['severity_level'].upper()
                st.markdown(f"**⚠️ Nivel de Severidad:** {severity_emoji.get(rec['severity_level'], '⚪')} **{severity_label}**")
            
            # Recomendaciones inmediatas
            st.markdown("### 🎯 Acciones Recomendadas")
            for item in rec['recommendations']:
                st.markdown(f"{item}")
            
            # Plan de tratamiento (si existe)
            if 'treatment_plan' in rec:
                with st.expander("📅 Plan de Tratamiento Detallado"):
                    for step in rec['treatment_plan']:
                        st.markdown(f"- {step}")
            
            # Tips preventivos
            if 'preventive_tips' in rec:
                with st.expander("💡 Consejos Preventivos"):
                    for tip in rec['preventive_tips']:
                        st.markdown(f"{tip}")
            
            # Análisis técnico
            with st.expander("🔧 Análisis Técnico Detallado"):
                st.write(f"**Clase predicha:** {st.session_state['prediction']}")
                st.write(f"**Confianza:** {conf*100:.2f}%")
                st.write(f"**Tiempo de inferencia:** {inf_time*1000:.2f} ms")
                
                st.markdown("**Características de la imagen:**")
                st.write(f"- Severidad estimada: {features['severity']:.1f}%")
                st.write(f"- Tonos marrones: {features['brown_percentage']:.1f}%")
                st.write(f"- Manchas oscuras: {features['dark_spots_percentage']:.1f}%")
                st.write(f"- Varianza de color: {features['color_variance']:.2f}")
                
                st.markdown("**Probabilidades por clase:**")
                for i, cls in enumerate(class_names):
                    prob = all_pred[i] * 100
                    st.write(f"- {cls}: {prob:.2f}%")
            
            # Botón para nueva predicción
            st.markdown("---")
            if st.button("🔄 Analizar otra imagen", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        else:
            # Estado inicial - Guía visual atractiva
            st.info("👆 **Carga una imagen en el panel izquierdo para comenzar el análisis**")
            
if __name__ == '__main__':
    main()