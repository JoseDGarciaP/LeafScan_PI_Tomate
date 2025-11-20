import io
import numpy as np


# en la Raspberry se usa tflite_runtime
try:
import tflite_runtime.interpreter as tflite
except Exception:
# para pruebas en PC si no está tflite-runtime
import tensorflow as tf
tflite = None


MODEL_PATH = 'model_quantized.tflite' # coloca en mismo directorio
CLASS_NAMES = 'class_names.json'


app = FastAPI()


# Cargar class_names
with open(CLASS_NAMES, 'r') as f:
class_names = json.load(f)


# Inicializar interpreter
if tflite is not None:
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
else:
# fallback: usar TensorFlow interpreter (solo para debug en PC)
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)


interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


INPUT_SHAPE = input_details[0]['shape']
IMG_SIZE = (INPUT_SHAPE[1], INPUT_SHAPE[2])




def preprocess_image(data: bytes):
img = Image.open(io.BytesIO(data)).convert('RGB')
img = img.resize(IMG_SIZE)
arr = np.array(img).astype(np.float32)
# MobileNetV2 preprocess: escala entre -1 y 1
arr = (arr / 127.5) - 1.0
arr = np.expand_dims(arr, axis=0)
return arr




@app.post('/predict')
async def predict(file: UploadFile = File(...)):
data = await file.read()
x = preprocess_image(data)


start = time.time()
interpreter.set_tensor(input_details[0]['index'], x)
interpreter.invoke()
preds = interpreter.get_tensor(output_details[0]['index'])
elapsed = time.time() - start


top_idx = int(np.argmax(preds[0]))
confidence = float(preds[0][top_idx])
class_name = class_names[top_idx]


return JSONResponse({
'class_id': top_idx,
'class_name': class_name,
'confidence': confidence,
'inference_time': elapsed
})