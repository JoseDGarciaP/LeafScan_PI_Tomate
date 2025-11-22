"""
convert_tflite.py - Convierte el modelo H5 a TFLite
"""
import tensorflow as tf
import numpy as np
from pathlib import Path
import os

def convert_to_tflite():
    print("="*60)
    print("CONVERSIÓN A TFLITE PARA RASPBERRY PI")
    print("="*60)
    
    # Intentar diferentes rutas
    possible_paths = [
        '../models/model_tomato_disease.h5',  # Desde src/
        'models/model_tomato_disease.h5',  # Desde raíz
    ]
    
    MODEL_H5 = None
    for path in possible_paths:
        if os.path.exists(path):
            MODEL_H5 = path
            break
    
    if MODEL_H5 is None:
        print("\n✗ Error: No se encontró el modelo")
        print("Buscado en:")
        for p in possible_paths:
            print(f"  - {p}")
        return False
    
    # Definir ruta de salida
    if MODEL_H5.startswith('..'):
        OUT_TFLITE = '../models/model_tomato_quantized.tflite'
    else:
        OUT_TFLITE = 'models/model_tomato_quantized.tflite'
    
    print(f"\n✓ Modelo encontrado: {MODEL_H5}")
    print("Cargando modelo...")
    
    try:
        # Cargar modelo con safe_mode=False para evitar errores de capas personalizadas
        model = tf.keras.models.load_model(MODEL_H5, compile=False, safe_mode=False)
        print("✓ Modelo cargado exitosamente")
    except Exception as e:
        print(f"\n✗ Error al cargar el modelo: {e}")
        print("\nIntentando método alternativo...")
        try:
            # Método alternativo: cargar sin compilar
            model = tf.keras.models.load_model(MODEL_H5, compile=False)
            print("✓ Modelo cargado con método alternativo")
        except Exception as e2:
            print(f"✗ También falló: {e2}")
            return False
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    print("\nConvirtiendo modelo a TFLite...")
    try:
        tflite_model = converter.convert()
        
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(OUT_TFLITE), exist_ok=True)
        
        with open(OUT_TFLITE, 'wb') as f:
            f.write(tflite_model)
        
        original_size = Path(MODEL_H5).stat().st_size / (1024 * 1024)
        tflite_size = len(tflite_model) / (1024 * 1024)
        
        print("\n" + "="*60)
        print("✓ CONVERSIÓN EXITOSA")
        print("="*60)
        print(f"\n✓ Modelo TFLite guardado en:")
        print(f"  {OUT_TFLITE}")
        print(f"\n📊 Estadísticas:")
        print(f"  Tamaño original (H5): {original_size:.2f} MB")
        print(f"  Tamaño TFLite: {tflite_size:.2f} MB")
        print(f"  Reducción: {((original_size - tflite_size) / original_size * 100):.1f}%")
        
        return True
        
    except Exception as e:
        print(f'\n✗ Error durante la conversión: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = convert_to_tflite()
    
    if success:
        print("\n" + "="*60)
        print("🎉 ¡LISTO PARA USAR!")
        print("="*60)
        print("\n📋 Próximo paso:")
        print("  cd ..")
        print("  streamlit run streamlit_app.py")
        print()
    else:
        print("\n❌ La conversión falló.")