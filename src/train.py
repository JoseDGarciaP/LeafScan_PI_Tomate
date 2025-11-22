"""
train.py - Entrenamiento del modelo para clasificación de hojas de tomate
Dataset: Tomato_healthy y Tomato_Bacterial_spot
"""
import argparse
import json
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import mobilenet_v2


def build_dataset(dataset_dir: str, image_size, batch_size, seed=123):
    """Carga dataset desde carpetas: subcarpetas = clases"""
    print(f"\n📂 Cargando dataset desde: {dataset_dir}")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset='training',
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset='validation',
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )
    
    class_names = train_ds.class_names
    print(f"✓ Clases encontradas: {class_names}")
    
    # Optimización del pipeline
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, class_names


def build_model(num_classes: int, input_shape=(224, 224, 3)):
    """Construye el modelo usando MobileNetV2 con Transfer Learning"""
    print("\n🏗️  Construyendo modelo MobileNetV2...")
    
    base_model = mobilenet_v2.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=input_shape)
    x = mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    print("✓ Modelo construido exitosamente")
    
    return model


def compute_class_weights(dataset_dir: str, class_names):
    """Calcula los pesos de clase para balancear el entrenamiento"""
    print("\n⚖️  Calculando pesos de clase...")
    
    counts = {}
    for i, cname in enumerate(class_names):
        p = Path(dataset_dir) / cname
        jpgs = list(p.glob('*.jpg')) + list(p.glob('*.jpeg')) + list(p.glob('*.png')) + list(p.glob('*.JPG'))
        counts[i] = len(jpgs)
        print(f"  Clase '{cname}': {len(jpgs)} imágenes")
    
    total = sum(counts.values())
    class_weights = {i: total / (len(class_names) * count) for i, count in counts.items()}
    
    print(f"✓ Pesos calculados: {class_weights}")
    return class_weights


def main():
    parser = argparse.ArgumentParser(description='Entrenamiento para detección de bacteria en hojas de tomate')
    parser.add_argument('--dataset_dir', type=str, default='PlantVillage',
                        help='Directorio con Tomato_healthy y Tomato_Bacterial_spot')
    parser.add_argument('--output_models', type=str, default='models',
                        help='Directorio de salida para modelos')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Épocas de entrenamiento inicial')
    parser.add_argument('--fine_tune_epochs', type=int, default=5,
                        help='Épocas de fine-tuning')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Tamaño del batch')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Tamaño de imagen')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Usar pesos de clase para balancear')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🍅 ENTRENAMIENTO - DETECCIÓN DE BACTERIA EN TOMATE")
    print("=" * 60)
    print(f"\nParámetros:")
    print(f"  Dataset: {args.dataset_dir}")
    print(f"  Epochs: {args.epochs} + {args.fine_tune_epochs} (fine-tuning)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Tamaño imagen: {args.img_size}x{args.img_size}")
    print(f"  Usar pesos de clase: {args.use_class_weights}")
    
    # Cargar dataset
    image_size = (args.img_size, args.img_size)
    train_ds, val_ds, class_names = build_dataset(args.dataset_dir, image_size, args.batch_size)
    
    # Construir modelo
    model = build_model(num_classes=len(class_names), input_shape=(args.img_size, args.img_size, 3))
    
    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n📊 Resumen del modelo:")
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor='val_accuracy',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Calcular pesos de clase si se solicita
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(args.dataset_dir, class_names)
    
    # FASE 1: Entrenamiento inicial
    print("\n" + "=" * 60)
    print("🚀 FASE 1: Entrenamiento inicial (capas congeladas)")
    print("=" * 60)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # FASE 2: Fine-tuning
    print("\n" + "=" * 60)
    print("🔧 FASE 2: Fine-tuning (descongelando últimas capas)")
    print("=" * 60)
    
    # Encontrar el base_model correctamente
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and 'mobilenet' in layer.name.lower():
            base_model = layer
            break
    
    if base_model is None:
        print("⚠️  No se encontró MobileNet, intentando con layers[2]...")
        try:
            base_model = model.layers[2]
        except:
            print("⚠️  No se puede hacer fine-tuning, guardando modelo actual...")
            base_model = None
    
    if base_model is not None:
        try:
            base_model.trainable = True
            
            # Congelar todas las capas excepto las últimas 20
            for layer in base_model.layers[:-20]:
                layer.trainable = False
            
            print(f"✓ Descongeladas las últimas 20 capas de {len(base_model.layers)} capas totales")
            
            # Recompilar con learning rate más bajo
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            history_fine = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.fine_tune_epochs,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
        except Exception as e:
            print(f"⚠️  Error durante fine-tuning: {e}")
            print("  Continuando sin fine-tuning...")
    
    # Guardar modelo
    print("\n" + "=" * 60)
    print("💾 GUARDANDO MODELO")
    print("=" * 60)
    
    os.makedirs(args.output_models, exist_ok=True)
    model_path = os.path.join(args.output_models, 'model_tomato_disease.h5')
    model.save(model_path)
    print(f"✓ Modelo guardado en: {model_path}")
    
    # Convertir a TFLite inmediatamente
    print("\n" + "="*60)
    print("🔄 CONVIRTIENDO A TFLITE")
    print("="*60)
    
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(args.output_models, 'model_tomato_quantized.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        h5_size = Path(model_path).stat().st_size / (1024 * 1024)
        tflite_size = len(tflite_model) / (1024 * 1024)
        
        print(f"✓ Modelo TFLite guardado en: {tflite_path}")
        print(f"  Tamaño H5: {h5_size:.2f} MB")
        print(f"  Tamaño TFLite: {tflite_size:.2f} MB")
        print(f"  Reducción: {((h5_size - tflite_size) / h5_size * 100):.1f}%")
        
    except Exception as e:
        print(f"⚠️  Error al convertir a TFLite: {e}")
        print("  Puedes intentar convertir manualmente después")
    
    # Guardar nombres de clases
    class_names_path = os.path.join(args.output_models, 'class_names.json')
    with open(class_names_path, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)
    print(f"✓ Clases guardadas en: {class_names_path}")
    
    # Evaluación final
    print("\n" + "=" * 60)
    print("📈 EVALUACIÓN FINAL")
    print("=" * 60)
    
    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"Pérdida de validación: {loss:.4f}")
    print(f"Precisión de validación: {accuracy*100:.2f}%")
    
    if accuracy >= 0.90:
        print("\n🎉 ¡Excelente! Precisión mayor al 90%")
    elif accuracy >= 0.80:
        print("\n✅ ¡Bien! Precisión mayor al 80%")
    elif accuracy >= 0.70:
        print("\n⚠️  Precisión aceptable, pero podría mejorar")
    else:
        print("\n❌ Precisión baja. Considera:")
        print("   - Agregar más imágenes")
        print("   - Entrenar por más epochs")
        print("   - Usar --use_class_weights")
    
    print("\n" + "=" * 60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print("\n📋 Próximos pasos:")
    print("  1. Probar la app: streamlit run streamlit_app.py")
    print()


if __name__ == '__main__':
    main()