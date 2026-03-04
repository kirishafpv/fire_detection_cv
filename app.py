import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

st.set_page_config(page_title="🔥 Fire Detection", layout="wide")

st.title("🔥 Fire Detection App")
st.markdown("Загрузите изображение, и модель определит, есть ли на нём пожар. 🌟")

# =======================
# Настройки устройства
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# Загрузка модели
# =======================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("fire_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

classes = ["fire 🔥", "no_fire ✅"]

# =======================
# Grad-CAM функции
# =======================
def grad_cam(model, img_tensor, target_class):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # регистрируем хук на последний слой conv
    target_layer = model.layer4[1].conv2
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    output.backward(gradient=one_hot, retain_graph=True)

    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= np.min(cam)
    cam /= np.max(cam) + 1e-8
    h1.remove()
    h2.remove()
    return cam

# =======================
# Загрузка изображения
# =======================
uploaded_file = st.file_uploader("📷 Загрузите изображение", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("### Исходное изображение")
    st.image(image, width=400)

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    prediction = classes[pred.item()]
    confidence = conf.item() * 100

    # =======================
    # Grad-CAM overlay
    # =======================
    cam = grad_cam(model, input_tensor, pred.item())
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_np = np.array(image.resize((224, 224)))
    overlay = np.uint8(0.5 * img_np + 0.5 * heatmap)

    st.markdown("### 🔥 Grad-CAM (области, где модель видит огонь)")
    st.image(overlay, width=400)

    # =======================
    # Карточка результата
    # =======================
    if pred.item() == 0:  # fire
        st.markdown(f"<div style='padding:20px; background-color:#ff4b4b; color:white; border-radius:15px;'>"
                    f"<h2>🔥 Предсказание: {prediction}</h2>"
                    f"<h3>Уверенность: {confidence:.2f}%</h3></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='padding:20px; background-color:#4CAF50; color:white; border-radius:15px;'>"
                    f"<h2>✅ Предсказание: {prediction}</h2>"
                    f"<h3>Уверенность: {confidence:.2f}%</h3></div>", unsafe_allow_html=True)

    # =======================
    # Вероятности классов
    # =======================
    st.markdown("### 📊 Вероятности классов")
    for i, cls in enumerate(classes):
        prob = probs[0, i].item()
        color = "#ff4b4b" if i == 0 else "#4CAF50"
        st.markdown(f"<div style='display:flex; justify-content:space-between;'>"
                    f"<span>{cls}</span><span>{prob*100:.2f}%</span></div>"
                    f"<div style='background-color:#ddd; border-radius:10px; width:100%; height:20px;'>"
                    f"<div style='width:{prob*100}%; background-color:{color}; height:20px; border-radius:10px;'></div>"
                    f"</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("💡 Красная зона = пожар 🔥, зелёная = безопасно ✅")
