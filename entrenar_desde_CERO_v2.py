"""
entrenar_desde_CERO_v2.py - Compatible con Keras 3 (TensorFlow 2.16+)
Guarda en formato .keras y genera TFLite optimizado
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import mobilenet_v2
import json
import os
import numpy as np
from pathlib import Path

print("="*70)
print("🔥 ENTRENAMIENTO DESDE CERO - KERAS 3 COMPATIBLE")
print("="*70)
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# LIMPIAR MODELOS ANTERIORES
models_to_clean = [
    'models/model_tomato_disease.h5',
    'models/model_tomato.keras',
    'models/model_tomato_quantized.tflite'
]

for model_path in models_to_clean:
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"🗑️  {model_path} eliminado")

# ORDEN EXPLÍCITO DE CLASES
class_names = ["Tomato_Bacterial_spot", "Tomato_healthy"]
print(f"\n📋 Orden de clases:")
print(f"  [0] = {class_names[0]}")
print(f"  [1] = {class_names[1]}")

# CARGAR DATOS
print("\n📂 Cargando datos...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    'PlantVillage',
    labels='inferred',
    label_mode='int',
    class_names=class_names,
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(224, 224),
    batch_size=32,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'PlantVillage',
    labels='inferred',
    label_mode='int',
    class_names=class_names,
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(224, 224),
    batch_size=32,
    shuffle=True
)

# Verificar datos
print("\n🔍 Verificando datos...")
for images, labels in train_ds.take(1):
    print(f"Batch shape: {images.shape}")
    print(f"Labels únicos: {np.unique(labels.numpy())}")
    print(f"Conteo por clase: {np.bincount(labels.numpy().astype(int))}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# CONSTRUIR MODELO
print("\n🏗️  Construyendo modelo...")

base_model = mobilenet_v2.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(2, activation='softmax', name='predictions')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Modelo construido")
model.summary()

# ENTRENAR
print("\n" + "="*70)
print("🚀 ENTRENAMIENTO (30 EPOCHS)")
print("="*70)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

# EVALUACIÓN
print("\n" + "="*70)
print("📊 EVALUACIÓN")
print("="*70)

loss, acc = model.evaluate(val_ds, verbose=0)
print(f"Precisión en validación: {acc*100:.2f}%")

# PRUEBA MANUAL
print("\n🧪 PRUEBA CON IMÁGENES DE VALIDACIÓN...")
correct = 0
total = 0

for images, labels in val_ds.take(1):
    predictions = model.predict(images, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    for i in range(min(10, len(labels))):
        true_label = int(labels[i])
        pred_label = int(pred_classes[i])
        confidence = predictions[i][pred_label] * 100
        
        correct_mark = "✓" if true_label == pred_label else "✗"
        print(f"{correct_mark} Real: {class_names[true_label]:30s} | "
              f"Pred: {class_names[pred_label]:30s} ({confidence:.1f}%)")
        
        if true_label == pred_label:
            correct += 1
        total += 1

print(f"\nPrecisión en muestra: {correct}/{total} = {correct/total*100:.1f}%")

# CREAR CARPETA models
os.makedirs('models', exist_ok=True)

# GUARDAR EN FORMATO .keras (Keras 3)
print("\n" + "="*70)
print("💾 GUARDANDO MODELO (.keras)")
print("="*70)

model.save('models/model_tomato.keras')
print("✓ Modelo guardado en: models/model_tomato.keras")

keras_size = Path('models/model_tomato.keras').stat().st_size / (1024 * 1024)
print(f"  Tamaño: {keras_size:.2f} MB")

# GUARDAR class_names.json
with open('models/class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)
print("✓ Clases guardadas en: models/class_names.json")

# CONVERTIR A TFLITE (para Raspberry Pi)
print("\n" + "="*70)
print("🔄 CONVIRTIENDO A TFLITE")
print("="*70)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Agregar dataset representativo para cuantización
def representative_dataset():
    for images, _ in train_ds.take(100):
        for i in range(images.shape[0]):
            img = tf.cast(images[i:i+1], tf.float32)
            yield [img]

converter.representative_dataset = representative_dataset

try:
    tflite_model = converter.convert()
    
    with open('models/model_tomato_quantized.tflite', 'wb') as f:
        f.write(tflite_model)
    
    tflite_size = len(tflite_model) / (1024 * 1024)
    reduction = ((keras_size - tflite_size) / keras_size * 100)
    
    print(f"✓ TFLite guardado: models/model_tomato_quantized.tflite")
    print(f"  Tamaño .keras: {keras_size:.2f} MB")
    print(f"  Tamaño TFLite: {tflite_size:.2f} MB")
    print(f"  Reducción: {reduction:.1f}%")
except Exception as e:
    print(f"⚠️  Error en conversión TFLite: {e}")
    print("El modelo .keras funciona, pero no se pudo generar TFLite")

print("\n" + "="*70)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*70)
print("\n🎯 Próximos pasos:")
print("  1. Probar en PC: streamlit run streamlit_app_v2.py")
print("  2. Si funciona bien, copiar archivos a Raspberry Pi:")
print("     - models/model_tomato_quantized.tflite")
print("     - models/class_names.json")
print("     - streamlit_app.py (el original con TFLite)")