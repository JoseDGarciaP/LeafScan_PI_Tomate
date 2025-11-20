import tensorflow as tf
import numpy as np
from pathlib import Path


MODEL_H5 = '../models/model_mobilenetv2.h5'
OUT_TFLITE = '../models/model_quantized.tflite'


# Conversión con cuantización post-training (entero si es posible)
model = tf.keras.models.load_model(MODEL_H5)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]


# Si quieres una cuantización completa (integer-only) se necesitará un dataset representativo.
# Aquí usamos optimizations DEFAULT que suele producir un buen modelo cuantizado por rango.


try:
tflite_model = converter.convert()
Path(OUT_TFLITE).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_TFLITE, 'wb') as f:
f.write(tflite_model)
print('Modelo TFLite guardado en', OUT_TFLITE)
except Exception as e:
print('Error al convertir:', e)