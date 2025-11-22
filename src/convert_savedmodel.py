"""
convert_savedmodel.py - Convierte usando SavedModel como intermediario
"""
import tensorflow as tf
import os
from pathlib import Path

print("="*60)
print("CONVERSIÓN VÍA SAVEDMODEL")
print("="*60)

# Paso 1: Cargar el H5 y guardarlo como SavedModel
print("\n📦 Paso 1: Cargando modelo H5...")

h5_path = '../models/model_tomato_disease.h5'
savedmodel_path = '../models/saved_model_temp'

try:
    # Cargar con compile=False para evitar problemas
    model = tf.keras.models.load_model(h5_path, compile=False)
    print("✓ Modelo H5 cargado (sin compilar)")
    
    # Guardarlo como SavedModel
    print("\n💾 Paso 2: Guardando como SavedModel...")
    model.save(savedmodel_path, save_format='tf')
    print(f"✓ SavedModel guardado en: {savedmodel_path}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("\n⚠️  Vamos a intentar otro método...")
    
    # Método alternativo: cargar arquitectura y pesos por separado
    print("\n🔄 Método alternativo: Reconstrucción inteligente...")
    
    import h5py
    import json
    
    # Leer la configuración del modelo del H5
    with h5py.File(h5_path, 'r') as f:
        # Obtener la configuración del modelo si existe
        if 'model_config' in f.attrs:
            model_config = json.loads(f.attrs['model_config'])
            print("✓ Configuración del modelo encontrada")
            
            # Reconstruir desde la configuración
            from tensorflow.keras.models import model_from_json
            model = model_from_json(json.dumps(model_config))
            
            # Cargar pesos
            model.load_weights(h5_path)
            print("✓ Modelo reconstruido y pesos cargados")
            
            # Guardar como SavedModel
            model.save(savedmodel_path, save_format='tf')
            print(f"✓ SavedModel guardado en: {savedmodel_path}")
        else:
            print("✗ No se pudo leer la configuración del modelo")
            exit(1)

# Paso 3: Convertir SavedModel a TFLite
print("\n🔄 Paso 3: Convirtiendo a TFLite...")

try:
    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    tflite_path = '../models/model_tomato_quantized.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    # Estadísticas
    h5_size = Path(h5_path).stat().st_size / (1024 * 1024)
    tflite_size = len(tflite_model) / (1024 * 1024)
    
    print("\n" + "="*60)
    print("✓ CONVERSIÓN EXITOSA")
    print("="*60)
    print(f"\n✓ Modelo TFLite guardado en: {tflite_path}")
    print(f"\n📊 Estadísticas:")
    print(f"  Tamaño H5: {h5_size:.2f} MB")
    print(f"  Tamaño TFLite: {tflite_size:.2f} MB")
    print(f"  Reducción: {((h5_size - tflite_size) / h5_size * 100):.1f}%")
    
    # Limpiar archivo temporal
    import shutil
    if os.path.exists(savedmodel_path):
        shutil.rmtree(savedmodel_path)
        print("\n🧹 Archivo temporal eliminado")
    
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