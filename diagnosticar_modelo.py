"""
diagnosticar_modelo.py - Verifica qué está prediciendo realmente el modelo
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from pathlib import Path

print("="*70)
print("🔍 DIAGNÓSTICO DEL MODELO")
print("="*70)

# Cargar modelo y clases
MODEL_PATH = 'models/model_tomato.keras'
CLASS_NAMES_PATH = 'models/class_names.json'

with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

print(f"\n📋 Clases en class_names.json:")
for i, cls in enumerate(class_names):
    print(f"  [{i}] = {cls}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"\n✓ Modelo cargado desde: {MODEL_PATH}")

# Función para probar con DIFERENTES preprocesamientos
def test_preprocessing_methods(image_path, class_names):
    """Prueba diferentes métodos de preprocesamiento"""
    image = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(image).astype(np.float32)
    
    print(f"\n{'='*70}")
    print(f"📸 Probando: {Path(image_path).name}")
    print(f"{'='*70}")
    
    # Método 1: MobileNetV2 preprocess_input (el del entrenamiento)
    print("\n1️⃣  MÉTODO: mobilenet_v2.preprocess_input (ENTRENAMIENTO)")
    img1 = tf.keras.applications.mobilenet_v2.preprocess_input(img_array.copy())
    img1 = np.expand_dims(img1, axis=0)
    pred1 = model.predict(img1, verbose=0)[0]
    idx1 = np.argmax(pred1)
    print(f"   Probabilidades: {pred1}")
    print(f"   Índice predicho: {idx1}")
    print(f"   Clase predicha: {class_names[idx1]}")
    print(f"   Confianza: {pred1[idx1]*100:.2f}%")
    
    # Método 2: Normalización [0, 1]
    print("\n2️⃣  MÉTODO: Normalización [0, 1]")
    img2 = img_array.copy() / 255.0
    img2 = np.expand_dims(img2, axis=0)
    pred2 = model.predict(img2, verbose=0)[0]
    idx2 = np.argmax(pred2)
    print(f"   Probabilidades: {pred2}")
    print(f"   Índice predicho: {idx2}")
    print(f"   Clase predicha: {class_names[idx2]}")
    print(f"   Confianza: {pred2[idx2]*100:.2f}%")
    
    # Método 3: Normalización [-1, 1]
    print("\n3️⃣  MÉTODO: Normalización [-1, 1]")
    img3 = (img_array.copy() / 127.5) - 1.0
    img3 = np.expand_dims(img3, axis=0)
    pred3 = model.predict(img3, verbose=0)[0]
    idx3 = np.argmax(pred3)
    print(f"   Probabilidades: {pred3}")
    print(f"   Índice predicho: {idx3}")
    print(f"   Clase predicha: {class_names[idx3]}")
    print(f"   Confianza: {pred3[idx3]*100:.2f}%")
    
    # Método 4: Sin normalización
    print("\n4️⃣  MÉTODO: Sin normalización [0, 255]")
    img4 = img_array.copy()
    img4 = np.expand_dims(img4, axis=0)
    pred4 = model.predict(img4, verbose=0)[0]
    idx4 = np.argmax(pred4)
    print(f"   Probabilidades: {pred4}")
    print(f"   Índice predicho: {idx4}")
    print(f"   Clase predicha: {class_names[idx4]}")
    print(f"   Confianza: {pred4[idx4]*100:.2f}%")

# Buscar imágenes de ejemplo
print("\n" + "="*70)
print("🔍 BUSCANDO IMÁGENES DE PRUEBA")
print("="*70)

bacterial_images = list(Path('PlantVillage/Tomato_Bacterial_spot').glob('*.jpg'))[:2]
healthy_images = list(Path('PlantVillage/Tomato_healthy').glob('*.jpg'))[:2]

print(f"\nEncontradas {len(bacterial_images)} imágenes con bacteria para probar")
print(f"Encontradas {len(healthy_images)} imágenes sanas para probar")

# Probar con imagen BACTERIAL
if bacterial_images:
    print("\n" + "🦠"*35)
    print("PROBANDO CON IMAGEN BACTERIAL")
    print("🦠"*35)
    test_preprocessing_methods(bacterial_images[0], class_names)

# Probar con imagen HEALTHY
if healthy_images:
    print("\n" + "🌿"*35)
    print("PROBANDO CON IMAGEN SANA")
    print("🌿"*35)
    test_preprocessing_methods(healthy_images[0], class_names)

print("\n" + "="*70)
print("✅ DIAGNÓSTICO COMPLETADO")
print("="*70)
print("\n💡 INTERPRETACIÓN:")
print("   - El MÉTODO 1 debe dar resultados correctos (es el del entrenamiento)")
print("   - Si MÉTODO 1 falla, hay problema con el modelo")
print("   - Si MÉTODO 1 funciona pero la app falla, hay problema en streamlit_app_v2.py")
print("="*70)