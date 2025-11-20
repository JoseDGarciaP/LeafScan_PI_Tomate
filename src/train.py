import argparse
import json
import os
from pathlib import Path


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import mobilenet_v2




def build_dataset(dataset_dir: str, image_size, batch_size, seed=123):
# Carga dataset desde carpetas: subcarpetas = clases
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


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


return train_ds, val_ds, class_names

def build_model(num_classes: int, input_shape=(224,224,3)):
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
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)


model = tf.keras.Model(inputs, outputs)
return model




def compute_class_weights(dataset_dir: str, class_names):
# cuenta archivos jpg/png por clase
counts = {}
for i, cname in enumerate(class_names):
p = Path(dataset_dir)/cname
jpgs = list(p.glob('*.jpg')) + list(p.glob('*.jpeg')) + list(p.glob('*.png'))
counts[i] = len(jpgs)
total = sum(counts.values())
class_weights = {i: total/(len(class_names)*count) for i, count in counts.items()}
return class_weights

def main():
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='PlantVillage')
parser.add_argument('--output_models', type=str, default='../models')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--fine_tune_epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--use_class_weights', action='store_true')
args = parser.parse_args()


image_size = (args.img_size, args.img_size)
train_ds, val_ds, class_names = build_dataset(args.dataset_dir, image_size, args.batch_size)


model = build_model(num_classes=len(class_names), input_shape=(args.img_size, args.img_size, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])


callbacks = [
tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy')
]


class_weights = compute_class_weights(args.dataset_dir, class_names) if args.use_class_weights else None


print('Entrenamiento inicial...')
model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, class_weight=class_weights)


# Fine-tuning: descongelar últimas capas del base_model
print('Fine-tuning...')
base_model = model.layers[2] # mobilenet preprocess + base_model -> asegurarse
try:
base_model.trainable = True
for layer in base_model.layers[:-20]:
layer.trainable = False
except Exception:
pass


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])


model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=callbacks, class_weight=class_weights)


os.makedirs(args.output_models, exist_ok=True)
model_path = os.path.join(args.output_models, 'model_mobilenetv2.h5')
model.save(model_path)
print(f'Modelo guardado en: {model_path}')


# Guardar classes
class_names_path = os.path.join(args.output_models, 'class_names.json')
with open(class_names_path, 'w') as f:
json.dump(class_names, f)
print(f'Clases guardadas en: {class_names_path}')




if __name__ == '__main__':
main()