"""
server_rpi.py - Servidor API REST para Raspberry Pi
Este servidor permite hacer predicciones mediante peticiones HTTP
Útil si quieres integrar con otras aplicaciones o servicios
"""
import io
import json
import time
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# Intentar usar tflite_runtime (Raspberry Pi)
try:
    import tflite_runtime.interpreter as tflite
    USING_TFLITE_RUNTIME = True
except ImportError:
    # Fallback a TensorFlow (PC)
    import tensorflow as tf
    tflite = None
    USING_TFLITE_RUNTIME = False

# Configuración
MODEL_PATH = 'models/model_tomato_quantized.tflite'
CLASS_NAMES_PATH = 'models/class_names.json'

# Inicializar FastAPI
app = FastAPI(
    title="Tomato Disease Detection API",
    description="API para detectar bacteria en hojas de tomate",
    version="1.0.0"
)

# Cargar nombres de clases
try:
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"✓ Clases cargadas: {class_names}")
except Exception as e:
    print(f"✗ Error al cargar class_names.json: {e}")
    class_names = ["Tomato_Bacterial_spot", "Tomato_healthy"]

# Inicializar intérprete TFLite
try:
    if USING_TFLITE_RUNTIME:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        print("✓ Usando tflite_runtime (Raspberry Pi)")
    else:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        print("✓ Usando TensorFlow Lite (PC)")
    
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    INPUT_SHAPE = input_details[0]['shape']
    IMG_SIZE = (INPUT_SHAPE[1], INPUT_SHAPE[2])
    
    print(f"✓ Modelo cargado: {MODEL_PATH}")
    print(f"  Tamaño de entrada: {IMG_SIZE}")
    
except Exception as e:
    print(f"✗ Error al cargar el modelo: {e}")
    print("  Asegúrate de tener el archivo model_tomato_quantized.tflite")
    interpreter = None


def preprocess_image(data: bytes):
    """Preprocesa la imagen para el modelo"""
    img = Image.open(io.BytesIO(data)).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    # MobileNetV2 preprocess: escala entre -1 y 1
    arr = (arr / 127.5) - 1.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.get("/")
async def root():
    """Endpoint raíz - información del servidor"""
    return {
        "message": "Tomato Disease Detection API",
        "version": "1.0.0",
        "status": "running",
        "runtime": "tflite_runtime" if USING_TFLITE_RUNTIME else "tensorflow",
        "classes": class_names,
        "endpoints": {
            "POST /predict": "Realizar predicción sobre una imagen",
            "GET /health": "Verificar estado del servidor"
        }
    }


@app.get("/health")
async def health():
    """Endpoint de salud"""
    return {
        "status": "healthy",
        "model_loaded": interpreter is not None,
        "runtime": "tflite_runtime" if USING_TFLITE_RUNTIME else "tensorflow"
    }


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """
    Endpoint principal de predicción
    
    Args:
        file: Imagen de hoja de tomate (JPG, PNG)
    
    Returns:
        JSON con la predicción, confianza y tiempo de inferencia
    """
    if interpreter is None:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Modelo no cargado",
                "message": "El modelo TFLite no se pudo cargar correctamente"
            }
        )
    
    try:
        # Leer imagen
        data = await file.read()
        
        # Preprocesar
        x = preprocess_image(data)
        
        # Inferencia
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])
        elapsed = time.time() - start
        
        # Obtener resultados
        top_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][top_idx])
        class_name = class_names[top_idx]
        
        # Determinar estado
        is_healthy = "healthy" in class_name.lower()
        status = "Hoja Sana" if is_healthy else "Bacteria Detectada"
        
        return JSONResponse({
            'success': True,
            'class_id': top_idx,
            'class_name': class_name,
            'confidence': confidence,
            'confidence_percent': f"{confidence * 100:.2f}%",
            'status': status,
            'inference_time_ms': f"{elapsed * 1000:.2f}",
            'all_predictions': {
                class_names[i]: float(preds[0][i]) 
                for i in range(len(class_names))
            }
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "Error al procesar la imagen"
            }
        )


if __name__ == '__main__':
    import uvicorn
    
    print("\n" + "="*60)
    print("SERVIDOR API - DETECTOR DE BACTERIA EN TOMATE")
    print("="*60)
    print(f"\nClases disponibles: {class_names}")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Runtime: {'tflite_runtime (RPi)' if USING_TFLITE_RUNTIME else 'TensorFlow (PC)'}")
    print("\nIniciando servidor en http://0.0.0.0:8000")
    print("Documentación en: http://0.0.0.0:8000/docs")
    print("\nPara detener: Ctrl+C")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)