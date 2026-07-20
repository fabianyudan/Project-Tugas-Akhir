import streamlit as st
import os
from pathlib import Path

st.set_page_config(page_title="ABSA Tokopedia Demo", layout="wide")

st.title("ABSA Tokopedia Demo")
st.caption("Demo ekstraksi aspek dan klasifikasi sentimen ulasan Tokopedia.")

ASPECT_MODEL_DIR = Path("absa_models/aspect_extractor")
SENTIMENT_MODEL_DIR = Path("absa_models/sentiment_classifier")

with st.expander("Cek status model"):
    st.write("Aspect folder ada:", ASPECT_MODEL_DIR.exists())
    st.write("Sentiment folder ada:", SENTIMENT_MODEL_DIR.exists())
    if ASPECT_MODEL_DIR.exists():
        st.write("Isi aspect folder:", os.listdir(ASPECT_MODEL_DIR))
    if SENTIMENT_MODEL_DIR.exists():
        st.write("Isi sentiment folder:", os.listdir(SENTIMENT_MODEL_DIR))

review = st.text_area(
    "Masukkan ulasan produk",
    "Barangnya bagus dan original, harga murah, tapi pengiriman sangat lama."
)

if st.button("Analisis Sentimen"):
    with st.spinner("Model sedang diproses..."):
        st.write("Nanti fungsi prediksi model kamu dipanggil di sini.")
