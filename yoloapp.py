import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ===== Настройки страницы =====
st.set_page_config(
    page_title="🔥 Fire Detection YOLO",
    page_icon="🔥",
    layout="centered"
)

# ===== Заголовок =====
st.title("🔥 Fire Detection with YOLO")
st.markdown("Загрузите изображение и модель обнаружит **огонь 🔥**")

# ===== Загрузка модели =====
@st.cache_resource
def load_model():
    model = YOLO("runs/detect/train3/weights/best.pt")
    return model

model = load_model()

# ===== Загрузка изображения =====
uploaded_file = st.file_uploader(
    "📷 Загрузите изображение",
    type=["jpg", "jpeg", "png"]
)

# ===== Обработка =====
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("📷 Загруженное изображение")
    st.image(image, width=500)

    if st.button("🔍 Найти огонь"):

        with st.spinner("Анализ изображения..."):

            results = model(image)

            # YOLO рисует bounding boxes
            annotated_image = results[0].plot()

        st.subheader("🔥 Результат детекции")

        st.image(
            annotated_image,
            caption="YOLO Detection",
            width=500
        )

        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:

            st.success("🔥 Обнаружен огонь!")

            for box in boxes:
                conf = float(box.conf[0])
                st.write(f"Confidence: **{conf*100:.2f}%**")

        else:
            st.info("Огонь не обнаружен ❄️")
