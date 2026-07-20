from pathlib import Path
import streamlit as st
import os

st.set_page_config(page_title="ABSA Tokopedia Demo", layout="wide")

st.title("✅ ABSA Tokopedia Demo Berhasil Kebuka")
st.write("Kalau tulisan ini muncul, berarti Streamlit normal.")
st.write("Current folder:", os.getcwd())
st.write("Isi folder root:", os.listdir("."))

st.write("Cek folder aspect:", os.path.exists("absa_models/aspect_extractor"))
st.write("Cek folder sentiment:", os.path.exists("absa_models/sentiment_classifier"))

st.stop()

BASE_MODEL_DIR = Path("absa_models")

ASPECT_MODEL_DIR = BASE_MODEL_DIR / "aspect_extractor"
SENTIMENT_MODEL_DIR = BASE_MODEL_DIR / "sentiment_classifier"
