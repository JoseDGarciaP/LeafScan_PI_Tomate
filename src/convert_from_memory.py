"""
convert_from_memory.py - Reconstruye y convierte el modelo
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import mobilenet_v2
import os
from pathlib import Path

print("="*60)
print("CONVERSIÓN ALTERNATIVA A TFLITE")
print("="*60)

# Reconstruir la arquitectura del modelo
print("\n🏗️  Reconstruyendo arquitectura del modelo...")

base_model = mobilenet_v2.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(2, activation='softmax')(x)  # 2 clases

model = tf.keras.Model(inputs, outputs)

# Cargar solo los pesos (no la arquitectura completa)
print("📦 Cargando pesos desde el archivo H5...")

try:
    model.load_weights('../models/model_tomato_disease.h5')
    print("✓ Pesos cargados exitosamente")
except Exception as e:
    print(f"✗ Error al cargar pesos: {e}")
    exit(1)

# Convertir a TFLite
print("\n🔄 Convirtiendo a TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

try:
    tflite_model = converter.convert()
    
    output_path = '../models/model_tomato_quantized.tflite'
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    h5_size = Path('../models/model_tomato_disease.h5').stat().st_size / (1024 * 1024)
    tflite_size = len(tflite_model) / (1024 * 1024)
    
    print("\n" + "="*60)
    print("✓ CONVERSIÓN EXITOSA")
    print("="*60)
    print(f"\n✓ Modelo TFLite guardado en: {output_path}")
    print(f"\n📊 Estadísticas:")
    print(f"  Tamaño H5: {h5_size:.2f} MB")
    print(f"  Tamaño TFLite: {tflite_size:.2f} MB")
    print(f"  Reducción: {((h5_size - tflite_size) / h5_size * 100):.1f}%")
    
    print("\n" + "="*60)
    print("🎉 ¡LISTO PARA USAR!")
    print("="*60)
    print("\n📋 Próximo paso:")
    print("  cd ..")
    print("  streamlit run streamlit_app.py")
    
except Exception as e:
    print(f"\n✗ Error durante conversión: {e}")
    import traceback
    traceback.print_exc()