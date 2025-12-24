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
