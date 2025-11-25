"""
streamlit_app_v3_moderno_corregido.py - Versión corregida con iconos SVG
"""
import streamlit as st
import numpy as np
from PIL import Image
import json
import time
import tensorflow as tf
import cv2

# ============================================================================
# CONFIGURACIÓN Y ESTILOS PERSONALIZADOS MEJORADOS
# ============================================================================

st.set_page_config(
    page_title="LeafScan Pi",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado mejorado con elementos visuales atractivos
st.markdown("""
<style>
    /* Importar Font Awesome */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Paleta de colores moderna y vibrante */
    :root {
        --primary-green: #10B981;
        --secondary-green: #059669;
        --accent-green: #34D399;
        --vibrant-teal: #06B6D4;
        --warning-yellow: #F59E0B;
        --warning-orange: #F97316;
        --error-red: #EF4444;
        --success-green: #10B981;
        --bg-dark: #0F172A;
        --card-bg: #1E293B;
        --card-hover: #334155;
    }
    
    /* Título principal con animación sutil */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .main-title {
        background: linear-gradient(135deg, #10B981 0%, #06B6D4 50%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        padding: 1rem 0;
        text-shadow: 0 0 40px rgba(16, 185, 129, 0.4);
        animation: fadeInUp 0.8s ease-out;
        text-align: center;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .main-subtitle {
        text-align: center;
        color: #94A3B8 !important;
        font-size: 1.3rem !important;
        margin-bottom: 2rem !important;
        animation: fadeInUp 1s ease-out;
    }
    
    /* Icon styles */
    .icon-large {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .icon-medium {
        font-size: 1.2rem;
        margin-right: 0.5rem;
    }
    
    .icon-small {
        font-size: 0.9rem;
        margin-right: 0.3rem;
    }
    
    /* Tarjetas modernas con hover effects */
    .modern-card {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        padding: 1.8rem;
        border-radius: 16px;
        border: 1px solid #334155;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        color: #E2E8F0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .modern-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #10B981, #06B6D4, #34D399);
    }
    
    .modern-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 16px 48px rgba(0,0,0,0.4);
        border-color: #10B981;
    }
    
    .success-card {
        background: linear-gradient(135deg, #064E3B 0%, #065F46 100%);
        border-color: #10B981;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #78350F 0%, #92400E 100%);
        border-color: #F59E0B;
    }
    
    .danger-card {
        background: linear-gradient(135deg, #7F1D1D 0%, #991B1B 100%);
        border-color: #EF4444;
    }
    
    /* Botones modernos con iconos */
    .stButton > button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.9rem 2.5rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        font-size: 1.1rem;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.6);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Métricas mejoradas */
    .stMetric {
        font-family: inherit;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10B981, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        color: #CBD5E1 !important;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Sidebar mejorado */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        border-right: 1px solid #334155;
    }
    
    .sidebar-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #334155;
        margin-bottom: 1rem;
    }
    
    /* Progress bar animada */
    .stProgress > div > div {
        background: linear-gradient(90deg, #10B981, #06B6D4, #34D399) !important;
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* Badges modernos */
    .modern-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.95rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        gap: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .modern-badge:hover {
        transform: scale(1.05);
    }
    
    .badge-success { 
        background: linear-gradient(135deg, #065F46 0%, #047857 100%);
        color: #D1FAE5;
        border: 2px solid #10B981;
    }
    .badge-warning { 
        background: linear-gradient(135deg, #92400E 0%, #B45309 100%);
        color: #FEF3C7;
        border: 2px solid #F59E0B;
    }
    .badge-danger { 
        background: linear-gradient(135deg, #991B1B 0%, #B91C1C 100%);
        color: #FEE2E2;
        border: 2px solid #EF4444;
    }
    
    /* Upload area mejorada */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%) !important;
        border: 2px dashed #10B981 !important;
        border-radius: 16px;
        padding: 3rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #34D399 !important;
        background: linear-gradient(135deg, #334155 0%, #475569 100%) !important;
        transform: translateY(-2px);
    }
    
    /* Iconos animados */
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
        color: #10B981;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    /* Grid de características */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .feature-item {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        border-radius: 12px;
        border: 1px solid #334155;
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        transform: translateY(-3px);
        border-color: #10B981;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    
    /* Corregir estilos de texto */
    .stMarkdown h3 {
        color: #10B981 !important;
        border-bottom: 2px solid #22C55E;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    
    .stMarkdown p {
        color: #E2E8F0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES DE CARGA Y ANÁLISIS
# ============================================================================

MODEL_PATH = 'models/model_tomato.keras'
CLASS_NAMES_PATH = 'models/class_names.json'

@st.cache_resource
def load_model():
    """Carga el modelo y las clases"""
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        return model, class_names
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        st.info("Ejecuta primero: `python entrenar_desde_CERO_v2.py`")
        return None, None

def analyze_image_features(image):
    """Analiza características visuales de la imagen"""
    img_array = np.array(image)
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    brown_mask = cv2.inRange(img_hsv, np.array([10, 50, 50]), np.array([30, 255, 255]))
    brown_percentage = (np.sum(brown_mask > 0) / brown_mask.size) * 100
    
    dark_green_mask = cv2.inRange(img_hsv, np.array([35, 40, 20]), np.array([85, 255, 120]))
    dark_spots_percentage = (np.sum(dark_green_mask > 0) / dark_green_mask.size) * 100
    
    color_variance = np.std(img_array)
    
    severity = min(100, brown_percentage * 2 + dark_spots_percentage * 1.5 + color_variance * 0.3)
    
    return {
        'brown_percentage': brown_percentage,
        'dark_spots_percentage': dark_spots_percentage,
        'color_variance': color_variance,
        'severity': severity
    }

def preprocess_image(image, target_size=(224, 224)):
    """Preprocesa la imagen"""
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
    """Genera recomendaciones dinámicas"""
    severity = features['severity']
    
    if "healthy" in class_name.lower():
        if confidence > 0.95:
            return {
                "status": "Hoja Completamente Sana",
                "color": "success",
                "badge": "badge-success",
                "icon": "fas fa-leaf",
                "confidence_level": "muy_alta",
                "description": "La hoja está en excelente estado. No se detectaron signos de enfermedad.",
                "recommendations": [
                    "Continuar con el plan de cuidados actual",
                    "Mantener monitoreo semanal preventivo",
                    "Registrar esta planta como referencia de salud",
                    "Asegurar condiciones ambientales óptimas"
                ],
                "preventive_tips": [
                    "**Riego:** Mantener humedad constante sin encharcamiento",
                    "**Luz:** 6-8 horas de sol directo diario",
                    "**Temperatura:** 21-27°C óptimo"
                ]
            }
        elif confidence > 0.80:
            return {
                "status": "Hoja Aparentemente Sana",
                "color": "success",
                "badge": "badge-success",
                "icon": "fas fa-seedling",
                "confidence_level": "alta",
                "description": "La hoja parece saludable, aunque se recomienda monitoreo.",
                "recommendations": [
                    "Realizar inspección visual detallada",
                    "Revisar el envés de la hoja cuidadosamente",
                    "Monitorear en los próximos 3-5 días",
                    "Comparar con otras hojas de la planta"
                ],
                "preventive_tips": [
                    "Verificar que no haya síntomas iniciales",
                    "Buscar pequeñas manchas o decoloración",
                    "Tomar foto de seguimiento"
                ]
            }
        else:
            return {
                "status": "Revisión Necesaria",
                "color": "warning",
                "badge": "badge-warning",
                "icon": "fas fa-search",
                "confidence_level": "media",
                "description": "El análisis no es concluyente. Se recomienda inspección manual.",
                "recommendations": [
                    "Inspeccionar la hoja manualmente con lupa",
                    "Tomar nuevas fotos con mejor iluminación",
                    "Consultar con un experto si hay dudas",
                    "Re-analizar en 24-48 horas"
                ],
                "preventive_tips": [
                    "Mejora la iluminación para nueva foto",
                    "Asegura que la hoja esté bien enfocada",
                    "Limpia la hoja suavemente antes de fotografiar"
                ]
            }
    else:
        if severity > 60 or confidence > 0.90:
            return {
                "status": "Bacteria Detectada - Acción Urgente",
                "color": "error",
                "badge": "badge-danger",
                "icon": "fas fa-exclamation-triangle",
                "confidence_level": "muy_alta",
                "severity_level": "alta" if severity > 60 else "moderada",
                "description": f"Se detectó mancha bacteriana con {confidence*100:.1f}% de confianza. La severidad estimada es {'ALTA' if severity > 60 else 'moderada'}.",
                "recommendations": [
                    "**URGENTE:** Aislar la planta inmediatamente",
                    "Remover todas las hojas afectadas con tijeras esterilizadas",
                    "Aplicar bactericida de cobre dentro de las 24 horas",
                    "Destruir (quemar) hojas infectadas - NO compostar",
                    "Desinfectar herramientas después del uso",
                    "Marcar plantas circundantes para monitoreo intensivo"
                ],
                "treatment_plan": [
                    "**Día 1:** Poda de hojas afectadas + Primera aplicación de bactericida",
                    "**Día 3:** Segunda aplicación de bactericida",
                    "**Día 7:** Tercera aplicación + Evaluación de progreso",
                    "**Día 14:** Inspección final y decisión de continuidad"
                ],
                "preventive_tips": [
                    "No regar por aspersión - solo riego por goteo",
                    "Mejorar ventilación entre plantas",
                    "Evitar trabajar con plantas cuando estén mojadas",
                    "Documentar evolución con fotos diarias"
                ]
            }
        elif severity > 30 or confidence > 0.75:
            return {
                "status": "Bacteria Detectada - Tratamiento Necesario",
                "color": "warning",
                "badge": "badge-warning",
                "icon": "fas fa-exclamation-circle",
                "confidence_level": "alta",
                "severity_level": "moderada",
                "description": f"Se detectó presencia bacteriana. Severidad estimada: MODERADA ({severity:.1f}%).",
                "recommendations": [
                    "Aislar la planta en las próximas 12 horas",
                    "Remover hojas visiblemente afectadas",
                    "Preparar tratamiento con bactericida cúprico",
                    "Inspeccionar plantas en un radio de 2 metros",
                    "Iniciar registro de tratamiento y seguimiento"
                ],
                "treatment_plan": [
                    "**Día 1-2:** Poda sanitaria + Primera aplicación",
                    "**Día 5:** Segunda aplicación preventiva",
                    "**Día 10:** Evaluación y tercera aplicación si es necesario",
                    "**Día 15:** Revisión y ajuste del plan"
                ],
                "preventive_tips": [
                    "Reducir humedad ambiental si es posible",
                    "Monitorear temperatura (> 27°C favorece bacteria)",
                    "Limitar manipulación de plantas hasta tratamiento",
                    "Fotografiar evolución cada 3 días"
                ]
            }
        else:
            return {
                "status": "Posible Bacteria - Monitoreo Cercano",
                "color": "warning",
                "badge": "badge-warning",
                "icon": "fas fa-microscope",
                "confidence_level": "media",
                "severity_level": "baja",
                "description": "Se detectaron indicios de bacteria. Requiere confirmación y monitoreo.",
                "recommendations": [
                    "Inspección manual detallada inmediata",
                    "Tomar fotos adicionales de diferentes ángulos",
                    "Re-analizar en 24 horas",
                    "Preparar para poda preventiva si empeora",
                    "Considerar consulta con agrónomo"
                ],
                "treatment_plan": [
                    "**Día 1:** Observación + Fotos de referencia",
                    "**Día 2:** Re-análisis con el sistema",
                    "**Día 3:** Decisión de tratamiento basada en evolución",
                    "**Continuar** monitoreo diario por 7 días"
                ],
                "preventive_tips": [
                    "Mejorar ventilación general del cultivo",
                    "Revisar prácticas de riego",
                    "Aumentar higiene de herramientas",
                    "Comparar con hojas de plantas vecinas",
                    "Capturar nueva imagen con mejor calidad"
                ]
            }

# ============================================================================
# INTERFAZ PRINCIPAL CORREGIDA
# ============================================================================

def render_sidebar(class_names):
    """Renderiza el sidebar moderno"""
    with st.sidebar:
        # Header del sidebar
        st.markdown("""
        <div class='sidebar-header'>
            <i class="fas fa-leaf icon-large" style="color: #10B981;"></i>
            <h2 style='color: #10B981; margin: 0;'>LeafScan Pi</h2>
            <p style='color: #94A3B8; margin: 0;'>Sistema Inteligente de Diagnóstico</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Información del sistema
        st.markdown("**Información del Sistema**")
        st.markdown("""
        <div class='modern-card'>
            <i class="fas fa-robot icon-small"></i> <strong>Modelo:</strong> MobileNetV2 Fine-tuned<br>
            <i class="fas fa-code-branch icon-small"></i> <strong>Versión:</strong> 1.0<br>
            <i class="fas fa-bullseye icon-small"></i> <strong>Precisión:</strong> ~95% en validación<br>
            <i class="fas fa-bolt icon-small"></i> <strong>Velocidad:</strong> Inferencia en tiempo real
        </div>
        """, unsafe_allow_html=True)
        
        # Características
        st.markdown("**Características**")
        st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
        
        features = [
            ("fas fa-search", "Análisis de Confianza"),
            ("fas fa-chart-bar", "Estimación de Severidad"), 
            ("fas fa-lightbulb", "Recomendaciones Dinámicas"),
            ("fas fa-bolt", "Procesamiento Rápido")
        ]
        
        for icon, title in features:
            st.markdown(f"""
            <div class='feature-item'>
                <i class='{icon} feature-icon'></i>
                <strong>{title}</strong>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Clases detectadas
        st.markdown("**Clases Detectadas**")
        for cls in class_names:
            icon = "fas fa-leaf" if "healthy" in cls.lower() else "fas fa-bug"
            cls_display = cls.replace('_', ' ').title()
            st.markdown(f"<i class='{icon} icon-small' style='color: #10B981;'></i> {cls_display}", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Guía rápida
        with st.expander("Guía para Mejores Fotos", expanded=False):
            st.success("**Sí hacer:**")
            st.markdown("""
            - Luz natural difusa
            - Imagen nítida
            - Hoja centrada
            - Fondo uniforme
            """)
            
            st.error("**No hacer:**")
            st.markdown("""
            - Fotos borrosas
            - Sombras fuertes
            - Hoja muy pequeña
            - Múltiples hojas
            """)
            
            st.info("**Mejor momento:**")
            st.markdown("""
            - Mañana: 8-10 AM
            - Tarde: 4-6 PM
            - Evitar: mediodía
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <p style='color: #64748B; font-size: 0.8rem; margin: 0;'>
                <strong style='color: #10B981;'>© 2025 LeafScan Pi</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

def render_results(rec, conf, features, inf_time, pred_class, all_pred, class_names):
    """Renderiza los resultados del análisis"""
    
    # Header con badge
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem 0;'>
        <span class='modern-badge {rec["badge"]}'>
            <i class='{rec["icon"]}'></i> {rec['status']}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Métricas principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Confianza",
            value=f"{conf*100:.1f}%",
            delta=rec['confidence_level'].replace('_', ' ').title()
        )
    
    with col2:
        speed_label = "Ultra Rápido" if inf_time < 0.3 else "Rápido"
        st.metric(
            label="Tiempo",
            value=f"{inf_time*1000:.0f} ms",
            delta=speed_label
        )
    
    with col3:
        if 'severity_level' in rec:
            st.metric(
                label="Severidad", 
                value=rec['severity_level'].title(),
                delta="Alta" if rec['severity_level'] == 'alta' else "Moderada"
            )
        else:
            st.metric(
                label="Estado",
                value="Saludable", 
                delta="Óptimo"
            )
    
    # Barra de confianza
    st.progress(conf)
    st.markdown("---")
    
    # Diagnóstico
    st.markdown("**Diagnóstico**")
    st.markdown(f"""
    <div class='modern-card {rec["color"]}-card'>
        {rec['description']}
    </div>
    """, unsafe_allow_html=True)
    
    # Recomendaciones
    st.markdown("**Acciones Recomendadas**")
    for i, item in enumerate(rec['recommendations'], 1):
        st.markdown(f"**{i}.** {item}")
    
    # Plan de tratamiento
    if 'treatment_plan' in rec:
        st.markdown("---")
        with st.expander("Plan de Tratamiento Detallado", expanded=True):
            for step in rec['treatment_plan']:
                st.markdown(f"• {step}")
    
    # Tips preventivos
    if 'preventive_tips' in rec:
        st.markdown("---")
        with st.expander("Consejos Preventivos", expanded=False):
            for tip in rec['preventive_tips']:
                st.markdown(f"• {tip}")

def main():
    """Función principal corregida"""
    model, class_names = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    render_sidebar(class_names)
    
    # Contenido principal
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # Título principal
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 class='main-title'>LeafScan Pi</h1>
            <p class='main-subtitle'>
                Sistema de diagnóstico inteligente para cultivos de tomate
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Área de carga
        st.markdown("**Cargar Imagen**")
        uploaded_file = st.file_uploader(
            "Selecciona una imagen de hoja de tomate",
            type=['jpg', 'jpeg', 'png'],
            help="Arrastra y suelta tu archivo aquí",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
            if st.button("Analizar Hoja", type="primary", use_container_width=True):
                with st.spinner("Analizando imagen..."):
                    features = analyze_image_features(image)
                    pred_class, conf, all_pred, inf_time = predict(model, class_names, image)
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
        else:
            # Placeholder cuando no hay imagen
            st.markdown("""
            <div class='modern-card' style='text-align: center; padding: 3rem 2rem;'>
                <i class="fas fa-leaf icon-large" style="color: #10B981; font-size: 4rem;"></i>
                <h3 style='color: #10B981;'>Comienza tu Análisis</h3>
                <p style='color: #E2E8F0;'>
                    Carga una imagen de hoja de tomate para obtener un diagnóstico completo
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Resultados del Análisis**")
        
        if 'prediction' in st.session_state:
            render_results(
                st.session_state['recommendation'],
                st.session_state['confidence'], 
                st.session_state['features'],
                st.session_state['inference_time'],
                st.session_state['prediction'],
                st.session_state['all_predictions'],
                class_names
            )
            
            st.markdown("---")
            if st.button("Analizar otra imagen", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        else:
            # Estado inicial
            st.markdown("""
            <div class='modern-card' style='text-align: center; padding: 4rem 2rem;'>
                <i class="fas fa-search icon-large" style="font-size: 4rem; color: #10B981;"></i>
                <h3 style='color: #10B981;'>Bienvenido a LeafScan Pi</h3>
                <p style='color: #E2E8F0;'>
                    Carga una imagen en el panel izquierdo para comenzar el análisis
                </p>
                <div style='background: #1E293B; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;'>
                    <p style='color: #94A3B8; margin: 0;'>
                        El sistema analizará automáticamente la salud de tu planta y te proporcionará recomendaciones específicas
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()