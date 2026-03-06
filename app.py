import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

st.set_page_config(page_title="🔥 Fire Detection", layout="wide")

st.title("🔥 Fire Detection System")
st.markdown("Система обнаружения пожара с использованием **ResNet18 + YOLOv8**")

# =======================
# Выбор модели
# =======================

model_type = st.sidebar.selectbox(
    "Выберите модель",
    ["ResNet18 (Classification)", "YOLOv8n (Detection fast easier)", "YOLOv8s (Detection better 'n')"]
)

# =======================
# Device
# =======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# Загрузка ResNet модели
# =======================

@st.cache_resource
def load_resnet():

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(torch.load("fire_model.pth", map_location=device))

    model.to(device)
    model.eval()

    return model


# =======================
# Загрузка YOLO
# =======================

@st.cache_resource
def load_yolo():

    model = YOLO("runs/detect/train3/weights/best.pt")

    return model


@st.cache_resource
def load_yolo_better():

    model = YOLO("runs/detect/train5/weights/best.torchscript")

    return model

# =======================
# Transform
# =======================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

classes = ["fire 🔥", "no_fire ✅"]

# =======================
# Grad-CAM
# =======================

def grad_cam(model, img_tensor, target_class):

    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.layer4[1].conv2

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)

    model.zero_grad()

    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1

    output.backward(gradient=one_hot)

    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grad, axis=(1, 2))

    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)

    cam = cv2.resize(cam, (224, 224))

    cam -= np.min(cam)
    cam /= np.max(cam) + 1e-8

    h1.remove()
    h2.remove()

    return cam


# =======================
# Upload image
# =======================

uploaded_file = st.file_uploader(
    "📷 Загрузите изображение",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("### 📷 Исходное изображение")
    st.image(image, width=400)

# =========================================================
# RESNET
# =========================================================

    if model_type == "ResNet18 (Classification)":

        model = load_resnet()

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)

            conf, pred = torch.max(probs, 1)

        prediction = classes[pred.item()]
        confidence = conf.item() * 100

        # Grad CAM
        cam = grad_cam(model, input_tensor, pred.item())

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        img_np = np.array(image.resize((224, 224)))

        overlay = np.uint8(0.5 * img_np + 0.5 * heatmap)

        st.markdown("### 🔥 Grad-CAM (где модель видит огонь)")
        st.image(overlay, width=400)

        # RESULT CARD

        if pred.item() == 0:

            st.markdown(
                f"""
                <div style='padding:20px;background:#ff4b4b;color:white;border-radius:15px'>
                <h2>🔥 Предсказание: {prediction}</h2>
                <h3>Уверенность: {confidence:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div style='padding:20px;background:#4CAF50;color:white;border-radius:15px'>
                <h2>✅ Предсказание: {prediction}</h2>
                <h3>Уверенность: {confidence:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        # probabilities

        st.markdown("### 📊 Вероятности классов")

        for i, cls in enumerate(classes):

            prob = probs[0, i].item()

            color = "#ff4b4b" if i == 0 else "#4CAF50"

            st.markdown(
                f"""
                <div style='display:flex;justify-content:space-between'>
                <span>{cls}</span>
                <span>{prob*100:.2f}%</span>
                </div>

                <div style='background:#ddd;border-radius:10px;width:100%;height:20px'>
                <div style='width:{prob*100}%;background:{color};height:20px;border-radius:10px'></div>
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================================================
# YOLO
# =========================================================

    if model_type == "YOLOv8n (Detection fast easier)":

        model = load_yolo()

        results = model(image)

        annotated = results[0].plot()

        st.markdown("### 🔥 YOLO Detection")
        st.image(annotated, width=500)

        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:

            st.success("🔥 Обнаружен огонь!")

            for box in boxes:

                conf = float(box.conf[0])

                st.write(f"Confidence: **{conf*100:.2f}%**")

        else:

            st.info("✅ Огонь не обнаружен")
    
    if model_type == "YOLOv8s (Detection better 'n')":

        model = load_yolo_better()

        results = model(image)

        annotated = results[0].plot()

        st.markdown("### 🔥 YOLO Detection")
        st.image(annotated, width=500)

        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:

            st.success("🔥 Обнаружен огонь!")

            for box in boxes:

                conf = float(box.conf[0])

                st.write(f"Confidence: **{conf*100:.2f}%**")

        else:

            st.info("✅ Огонь не обнаружен")
