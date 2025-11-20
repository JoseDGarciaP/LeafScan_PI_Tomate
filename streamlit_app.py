import streamlit as st
import requests
import io
from PIL import Image

BACKEND_URL = "http://192.168.1.50:5000/predict"   # Cambiar por la IP de la Raspberry

st.title("Clasificador de Enfermedades en Plantas — PlantVillage")
st.write("Sube una imagen de una hoja y el sistema detectará la enfermedad.")

uploaded_file = st.file_uploader("Subir imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    if st.button("Clasificar"):
        files = {"file": ("image.png", img_bytes, "image/png")}

        with st.spinner("Analizando..."):
            response = requests.post(BACKEND_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            st.success(f"**Resultado:** {result['label']} — {result['confidence'] * 100:.2f}%")
        else:
            st.error("Error al procesar la imagen.")
