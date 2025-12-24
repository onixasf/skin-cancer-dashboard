import streamlit as st
import numpy as np
import json
from PIL import Image
import plotly.express as px
import os
import tensorflow as tf
import requests

# ============================================================
# CONFIG
# ============================================================
MODEL_URL = "https://huggingface.co/onixasf/skin-cancer-efficientnetb3/resolve/main/best_efficientnet.keras"
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
# DOWNLOAD MODEL FROM HUGGINGFACE
# ============================================================
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

download_model()

# ============================================================
# LOAD MODEL & CLASS NAMES
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
