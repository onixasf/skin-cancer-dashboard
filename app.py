import requests
import streamlit as st
import numpy as np
import json
from PIL import Image
import plotly.express as px
import os
import tensorflow as tf

# ============================================================
# CONFIG
# ============================================================
MODEL_URL = "https://huggingface.co/spaces/onixasf/skin-cancer-efficientNetB3/resolve/main/best_efficientnet.keras"
MODEL_PATH = "best_efficientnet.keras"
CLASS_PATH = "class_names.json"
IMG_SIZE = (300, 300)
EDA_DIR = "pics"

st.set_page_config(
    page_title="Skin Cancer Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
html, body { font-family: 'Segoe UI', sans-serif; }
.main-title { font-size: 44px; font-weight: 800; color: ##405d7a; text-align: center; margin-top: -20px; }
.sub-title { font-size: 20px; color: #566573; text-align: center; margin-bottom: 30px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DOWNLOAD MODEL FROM HUGGINGFACE
# ============================================================
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            r = requests.get(MODEL_URL)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

download_model()

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_class_names():
    with open(CLASS_PATH, "r") as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()

# ============================================================
# PAGE SELECTION
# ============================================================
st.markdown("<div class='main-title'>🩺 Skin Cancer Classification Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>EfficientNetB3 — HAM10000 Dataset — Tugas Besar Sains Data</div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Dataset Overview",
    "📊 EDA Visualization",
    "📈 Model Performance",
    "🩺 Try Image Prediction"
])

# ============================================================
# PAGE 1
# ============================================================
with tab1:
    st.header("📌 Dataset Overview")
    st.write("""
    Dashboard ini menggunakan **dataset HAM10000**, berisi 7 jenis klasifikasi lesi kulit.
    Pada proyek ini, dikembangkan sistem klasifikasi lengkap mulai dari eksplorasi dan pembersihan data,
    pelatihan model menggunakan EfficientNetB3, hingga implementasi dashboard interaktif berbasis Streamlit.
    """)

    st.markdown("### 🧬 Daftar Kelas Diagnosis")
    class_details = {
        "akiec": "Actinic Keratoses",
        "bcc":   "Basal Cell Carcinoma",
        "bkl":   "Benign Keratosis",
        "df":    "Dermatofibroma",
        "mel":   "Melanoma",
        "nv":    "Melanocytic Nevi",
        "vasc":  "Vascular Lesions",
    }

    list_html = "<ul>"
    for code, full in class_details.items():
        list_html += f"<li><b>{code}</b> — {full}</li>"
    list_html += "</ul>"
    st.markdown(list_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ℹ️ Informasi Model")
    st.write("""
    - Arsitektur: EfficientNetB3  
    - Input Size: 300×300  
    - Fine-tuning + Transfer Learning  
    - Optimizer: Adam  
    - Loss: Categorical Crossentropy  
    """)

# ============================================================
# PAGE 2
# ============================================================
with tab2:
    st.header("📊 EDA Visualization")

    def show_img(title, filename):
        st.markdown(f"### {title}")
        st.image(
            os.path.join(EDA_DIR, filename),
            use_container_width=True
        )

    # ================================
    # Baris 1 – Grafik Statistik
    # ================================
    col1, col2 = st.columns([1, 1])

    with col1:
        show_img("🔥 Jumlah Outlier vs Normal", "jumlah_outlier.png")
        show_img("🎯 Distribusi Umur Pasien", "Age_distribution.png")

    with col2:
        show_img("🔥 Correlation Heatmap", "correlation_heatmap.png")
        show_img("📌 Distribusi Kelas Dataset", "class_distribution.png")

    st.divider()

    # ================================
    # Baris 2 – Samples Overview (diperkecil)
    # ================================
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        show_img("🖼 Samples Overview", "samples_overview.png")

# ============================================================
# PAGE 3
# ============================================================
with tab3:
    st.header("📈 Model Performance")

    def show_img(title, filename):
        st.markdown(f"### {title}")
        st.image(
            os.path.join(EDA_DIR, filename),
            use_container_width=True
        )

    col1, col2 = st.columns([1, 1])  # kiri–kanan seimbang

    with col1:
        show_img("📈 Kurva Training Model", "Train_curve.png")

    with col2:
        show_img("🧩 Confusion Matrix", "confusion_matrix.png")

    st.info("""
    Model **EfficientNetB3** menunjukkan peningkatan performa signifikan dibanding CNN baseline.
    Model stabil pada validasi akhir dan bekerja baik di seluruh kelas setelah balancing dataset.
    """)

# ============================================================
# PAGE 4
# ============================================================
with tab4:
    st.header("🩺 Try Image Prediction")
    uploaded_file = st.file_uploader(
        "Upload gambar skin lesion (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    def preprocess(image):
        img = image.resize(IMG_SIZE)
        return np.expand_dims(np.array(img) / 255.0, 0)

    def predict(image):
        x = preprocess(image)
        probs = model.predict(x)[0]
        sorted_idx = probs.argsort()[::-1]
        return probs, sorted_idx

    if uploaded_file:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 📷 Preview Image")
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, use_container_width=True)

        with col2:
            st.markdown("### 🔍 Prediction Result")
            probs, idx = predict(img)
            top_class = class_names[idx[0]]
            top_prob = probs[idx[0]]
            st.success(f"Prediksi: **{top_class.upper()}** ({top_prob:.4f})")

            fig = px.bar(
                x=class_names,
                y=probs,
                labels={"x": "Kelas", "y": "Probabilitas"},
                title="Probability per Class",
                color=probs,
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SIDEBAR FOOTER
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    🩺 **Skin Cancer Classification Dashboard**  

    👩🏻‍💻 *Developed by*  
    **Onixa Shafa Putri Wibowo**  
    *1227050107*  

    📘 *Final Project – Sains Data*  
    """
)
