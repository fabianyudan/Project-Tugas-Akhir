import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# ============================================================
# KONFIGURASI STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Analisis Sentimen Berbasis Aspek pada Review Tokopedia",
    page_icon="🛒",
    layout="wide",
)

BASE_MODEL_DIR = Path("absa_models")
ASPECT_MODEL_DIR = BASE_MODEL_DIR / "aspect_extractor"
SENTIMENT_MODEL_DIR = BASE_MODEL_DIR / "sentiment_classifier"

# Fallback kalau kamu masih pakai nama folder lama
if not ASPECT_MODEL_DIR.exists() and Path("indobert_bilstm_crf_aspect").exists():
    ASPECT_MODEL_DIR = Path("indobert_bilstm_crf_aspect")

if not SENTIMENT_MODEL_DIR.exists() and Path("deberta_v3_aspect_sentiment").exists():
    SENTIMENT_MODEL_DIR = Path("deberta_v3_aspect_sentiment")

MAX_LENGTH_AE = 64
MAX_LENGTH_ASC = 160
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LEXICON ASPEK DAN SENTIMEN
# Dipakai sebagai fallback dan koreksi konteks lokal.
# ============================================================
ASPECT_LEXICON = {
    "quality": [
        "barang", "produk", "kualitas", "mutu", "bahan", "fungsi", "ukuran",
        "warna", "kondisi", "deskripsi", "original", "ori", "bagus", "jelek",
        "rusak", "cacat", "palsu", "awet", "sesuai"
    ],
    "price": [
        "harga", "biaya", "ongkir", "diskon", "promo", "murah", "mahal",
        "kemahalan", "worth", "terjangkau", "overprice"
    ],
    "delivery": [
        "pengiriman", "kirim", "dikirim", "kurir", "paket", "resi", "sampai",
        "datang", "cepat", "lama", "telat", "lambat", "terlambat"
    ],
    "service": [
        "seller", "penjual", "toko", "admin", "pelayanan", "layanan",
        "respon", "respons", "chat", "ramah", "responsif", "slow", "fast", "jutek"
    ],
    "packaging": [
        "packing", "packaging", "kemasan", "bungkus", "bubble", "bubblewrap",
        "dus", "rapi", "aman", "penyok", "sobek", "hancur", "tebal"
    ],
}

ASPECT_ORDER = ["delivery", "quality", "price", "service", "packaging", "general"]

POSITIVE_WORDS = {
    "bagus", "baik", "mantap", "puas", "suka", "recommended", "rekomen",
    "original", "ori", "murah", "cepat", "rapi", "aman", "ramah",
    "responsif", "sesuai", "awet", "worth", "terjangkau", "oke", "ok",
    "top", "keren", "memuaskan"
}

NEGATIVE_WORDS = {
    "jelek", "buruk", "kecewa", "rusak", "cacat", "palsu", "mahal",
    "lama", "telat", "lambat", "terlambat", "penyok", "sobek", "hancur",
    "jutek", "slow", "tidak", "nggak", "ga", "gak", "kurang", "parah",
    "mengecewakan", "bau", "zonk"
}

NEGATION_WORDS = {"tidak", "nggak", "ga", "gak", "bukan", "kurang"}


def normalize_token(token):
    token = str(token).lower().strip()
    token = re.sub(r"[^a-zA-Z0-9_]+", "", token)
    return token


def simple_tokenize(text):
    raw_tokens = re.findall(r"\b\w+\b", str(text).lower(), flags=re.UNICODE)
    return [normalize_token(tok) for tok in raw_tokens if normalize_token(tok)]


def find_one_aspect_match(tokens, aspect_lexicon):
    """Cari aspek pertama yang muncul di teks."""
    best = None
    for idx, tok in enumerate(tokens):
        for category, terms in aspect_lexicon.items():
            terms_norm = {normalize_token(t) for t in terms}
            if tok in terms_norm:
                candidate = {
                    "aspect_text": tok,
                    "aspect_category": category,
                    "position": idx,
                }
                if best is None or candidate["position"] < best["position"]:
                    best = candidate
    return best


def score_lexicon_sentiment(tokens, selected_aspect=None, window=4):
    """Prediksi sentimen sederhana berdasarkan kata sekitar aspek."""
    if not tokens:
        return "neutral"

    if selected_aspect is not None:
        pos = int(selected_aspect.get("position", 0))
        start = max(0, pos - window)
        end = min(len(tokens), pos + window + 1)
        context_tokens = tokens[start:end]
    else:
        context_tokens = tokens

    score = 0
    for i, tok in enumerate(context_tokens):
        if tok in POSITIVE_WORDS:
            # Jika ada negasi sebelum kata positif, balik jadi negatif
            prev = set(context_tokens[max(0, i - 2):i])
            score += -1 if prev.intersection(NEGATION_WORDS) else 1

        if tok in NEGATIVE_WORDS:
            # Kata negasi sendiri jangan selalu dihitung negatif jika berdiri sendiri
            if tok in NEGATION_WORDS:
                continue
            prev = set(context_tokens[max(0, i - 2):i])
            score += 1 if prev.intersection(NEGATION_WORDS) else -1

    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


# ============================================================
# MODEL IndoBERT-BiLSTM-CRF
# Harus sama dengan class saat training di notebook.
# ============================================================
try:
    from torchcrf import CRF
except Exception:
    from TorchCRF import CRF


class IndoBERTBiLSTMCRF(nn.Module):
    def __init__(self, model_name, num_labels, lstm_hidden=64, lstm_layers=1, dropout=0.3, freeze_bert=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)

        # Umumnya memakai package pytorch-crf: CRF(num_tags, batch_first=True)
        try:
            self.crf = CRF(num_labels, batch_first=True)
            self._crf_style = "pytorch-crf"
        except TypeError:
            self.crf = CRF(num_labels)
            self._crf_style = "torchcrf-alt"

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        lstm_output, _ = self.bilstm(sequence_output)
        emissions = self.classifier(self.dropout(lstm_output))
        mask = attention_mask.bool()

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
            return loss, emissions

        if hasattr(self.crf, "decode"):
            return self.crf.decode(emissions, mask=mask)

        if hasattr(self.crf, "viterbi_decode"):
            return self.crf.viterbi_decode(emissions, mask)

        raise RuntimeError("CRF decode method tidak ditemukan. Cek package pytorch-crf di requirements.txt.")


def torch_load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_id2label(id2label):
    fixed = {}
    for k, v in id2label.items():
        try:
            fixed[int(k)] = v
        except Exception:
            fixed[k] = v
    return fixed


@st.cache_resource(show_spinner=False)
def load_absa_models():
    """Load dua model. Cache agar tidak reload setiap klik."""
    if not ASPECT_MODEL_DIR.exists():
        raise FileNotFoundError(f"Folder model aspek tidak ditemukan: {ASPECT_MODEL_DIR}")

    if not SENTIMENT_MODEL_DIR.exists():
        raise FileNotFoundError(f"Folder model sentimen tidak ditemukan: {SENTIMENT_MODEL_DIR}")

    aspect_ckpt_path = ASPECT_MODEL_DIR / "model.pt"
    if not aspect_ckpt_path.exists():
        raise FileNotFoundError(f"File model.pt tidak ditemukan: {aspect_ckpt_path}")

    checkpoint = torch_load_checkpoint(aspect_ckpt_path, DEVICE)

    label2id = checkpoint.get("label2id", {"O": 0, "B-ASP": 1, "I-ASP": 2})
    id2label = checkpoint.get("id2label", {0: "O", 1: "B-ASP", 2: "I-ASP"})
    id2label = normalize_id2label(id2label)

    model_name = checkpoint.get("model_name", "indobenchmark/indobert-base-p1")

    try:
        aspect_tokenizer = AutoTokenizer.from_pretrained(str(ASPECT_MODEL_DIR))
    except Exception:
        aspect_tokenizer = AutoTokenizer.from_pretrained(model_name)

    aspect_model = IndoBERTBiLSTMCRF(
        model_name=model_name,
        num_labels=len(label2id),
        lstm_hidden=64,
        dropout=0.3,
        freeze_bert=False,
    )
    aspect_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    aspect_model.to(DEVICE)
    aspect_model.eval()

    sentiment_tokenizer = AutoTokenizer.from_pretrained(str(SENTIMENT_MODEL_DIR))
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(str(SENTIMENT_MODEL_DIR))
    sentiment_model.to(DEVICE)
    sentiment_model.eval()

    sentiment_id2label = normalize_id2label(sentiment_model.config.id2label)
    if set(sentiment_id2label.values()) == {"LABEL_0", "LABEL_1", "LABEL_2"}:
        sentiment_id2label = {0: "negative", 1: "neutral", 2: "positive"}

    return {
        "aspect_model": aspect_model,
        "aspect_tokenizer": aspect_tokenizer,
        "label2id": label2id,
        "id2label": id2label,
        "sentiment_model": sentiment_model,
        "sentiment_tokenizer": sentiment_tokenizer,
        "sentiment_id2label": sentiment_id2label,
    }


def merge_aspect_spans(tokens, labels):
    spans = []
    current = []

    for tok, lab in zip(tokens, labels):
        if lab == "B-ASP":
            if current:
                spans.append(" ".join(current))
            current = [tok]
        elif lab == "I-ASP" and current:
            current.append(tok)
        else:
            if current:
                spans.append(" ".join(current))
                current = []

    if current:
        spans.append(" ".join(current))

    return spans


def predict_aspects_with_model(text, models, top_k=1):
    aspect_model = models["aspect_model"]
    aspect_tokenizer = models["aspect_tokenizer"]
    id2label = models["id2label"]

    tokens = simple_tokenize(text)

    if not tokens:
        return [], []

    encoding = aspect_tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH_AE,
        return_tensors="pt",
    )

    word_ids = encoding.word_ids(batch_index=0)
    encoding_device = {k: v.to(DEVICE) for k, v in encoding.items()}

    with torch.no_grad():
        pred_ids = aspect_model(**encoding_device)[0]

    word_label_map = {}
    for token_pos, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx not in word_label_map and token_pos < len(pred_ids):
            word_label_map[word_idx] = id2label.get(int(pred_ids[token_pos]), "O")

    word_labels = [word_label_map.get(i, "O") for i in range(len(tokens))]
    aspects = merge_aspect_spans(tokens, word_labels)

    if top_k is not None:
        aspects = aspects[:top_k]

    return aspects, list(zip(tokens, word_labels))


def map_raw_aspect_to_category(review_text, raw_aspects=None):
    if raw_aspects:
        for raw_asp in raw_aspects[:1]:
            raw_tokens = simple_tokenize(raw_asp)
            raw_joined = " ".join(raw_tokens)

            for category in ASPECT_ORDER:
                if category == "general":
                    continue

                terms_norm = {normalize_token(t) for t in ASPECT_LEXICON.get(category, [])}
                if any(t in terms_norm for t in raw_tokens) or raw_joined in terms_norm:
                    return category

    tokens = simple_tokenize(review_text)
    selected = find_one_aspect_match(tokens, ASPECT_LEXICON)
    if selected is not None:
        return selected["aspect_category"]

    return "general"


def local_context_sentiment(review_text, aspect_category, window=4):
    tokens = simple_tokenize(review_text)

    selected = None
    if aspect_category != "general":
        selected = find_one_aspect_match(
            tokens,
            {aspect_category: ASPECT_LEXICON.get(aspect_category, [])}
        )

    return score_lexicon_sentiment(tokens, selected, window=window)


def predict_sentiment_with_model(review_text, aspect_category, models):
    sentiment_model = models["sentiment_model"]
    sentiment_tokenizer = models["sentiment_tokenizer"]
    sentiment_id2label = models["sentiment_id2label"]

    input_text = f"aspek: {aspect_category} ulasan: {review_text}"

    enc = sentiment_tokenizer(
        [input_text],
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH_ASC,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = sentiment_model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        pred_id = int(np.argmax(probs))

    model_sentiment = sentiment_id2label.get(pred_id, str(pred_id))
    confidence = float(probs[pred_id])

    return model_sentiment, confidence, probs


def predict_absa(review_text):
    models = load_absa_models()

    raw_aspects, token_labels = predict_aspects_with_model(review_text, models, top_k=1)
    aspect_category = map_raw_aspect_to_category(review_text, raw_aspects)

    model_sentiment, confidence, probs = predict_sentiment_with_model(
        review_text,
        aspect_category,
        models,
    )

    rule_sentiment = local_context_sentiment(review_text, aspect_category)

    if rule_sentiment is not None and rule_sentiment != "neutral":
        final_sentiment = rule_sentiment
        note = "Dikoreksi memakai lexicon konteks lokal"
    else:
        final_sentiment = model_sentiment
        note = "Prediksi model mDeBERTa-v3"

    result = {
        "review": review_text,
        "aspect": aspect_category,
        "sentiment": final_sentiment,
        "confidence": round(confidence, 4),
        "sentiment_model": model_sentiment,
        "raw_aspects_from_AE": ", ".join(raw_aspects) if raw_aspects else "-",
        "note": note,
    }

    return result, token_labels, probs, models["sentiment_id2label"]


def predict_fallback_lexicon(review_text):
    tokens = simple_tokenize(review_text)
    selected = find_one_aspect_match(tokens, ASPECT_LEXICON)

    if selected is None:
        aspect_category = "general"
    else:
        aspect_category = selected["aspect_category"]

    sentiment = local_context_sentiment(review_text, aspect_category)
    return {
        "review": review_text,
        "aspect": aspect_category,
        "sentiment": sentiment,
        "confidence": 0.0,
        "sentiment_model": "-",
        "raw_aspects_from_AE": "-",
        "note": "Fallback lexicon karena model belum berhasil dimuat",
    }, [(tok, "O") for tok in tokens]


# ============================================================
# UI
# ============================================================
st.title("🛒 Analisis Sentimen Berbasis Aspek pada Review Tokopedia Menggunakan IndoBERT")
st.caption("Pipeline: IndoBERT-BiLSTM-CRF untuk ekstraksi aspek + mDeBERTa-v3 untuk klasifikasi sentimen aspek.")

with st.expander("Status folder model", expanded=False):
    st.write("Device:", str(DEVICE))
    st.write("Aspect model dir:", str(ASPECT_MODEL_DIR), "✅" if ASPECT_MODEL_DIR.exists() else "❌")
    st.write("Sentiment model dir:", str(SENTIMENT_MODEL_DIR), "✅" if SENTIMENT_MODEL_DIR.exists() else "❌")

    if ASPECT_MODEL_DIR.exists():
        st.write("Isi folder aspect_extractor:")
        st.code("\n".join(os.listdir(ASPECT_MODEL_DIR)))

    if SENTIMENT_MODEL_DIR.exists():
        st.write("Isi folder sentiment_classifier:")
        st.code("\n".join(os.listdir(SENTIMENT_MODEL_DIR)))

examples = [
    "Barangnya bagus dan original, harga murah, tapi pengiriman sangat lama.",
    "Packing rapi dan aman, tetapi barangnya rusak saat sampai.",
    "Seller ramah dan respon cepat, tapi harga agak mahal.",
    "Produk sesuai deskripsi, kualitas bagus, pengiriman cepat.",
    "Barang datang telat dan kemasan penyok.",
]

selected_example = st.selectbox("Pilih contoh ulasan", examples)
review_text = st.text_area("Masukkan ulasan produk Tokopedia", value=selected_example, height=130)

col1, col2 = st.columns([1, 3])
with col1:
    analyze = st.button("Analisis", type="primary", use_container_width=True)

with col2:
    st.info("Output dibatasi menjadi 1 aspek utama per review")

if analyze:
    if not review_text.strip():
        st.warning("Masukkan teks ulasan terlebih dahulu.")
    else:
        with st.spinner("Memuat model dan melakukan prediksi..."):
            try:
                result, token_labels, probs, id2label_sentiment = predict_absa(review_text)
                model_loaded = True
            except Exception as e:
                st.error("Model gagal dimuat/dijalankan. App memakai fallback lexicon sementara.")
                st.exception(e)
                result, token_labels = predict_fallback_lexicon(review_text)
                probs = None
                id2label_sentiment = None
                model_loaded = False

        st.subheader("Hasil Prediksi")

        result_df = pd.DataFrame([{
            "Aspek": result["aspect"],
            "Sentimen Akhir": result["sentiment"],
            "Confidence Model": result["confidence"],
            "Sentimen Model": result["sentiment_model"],
            "Raw Aspect AE": result["raw_aspects_from_AE"],
            "Keterangan": result["note"],
        }])

        st.dataframe(result_df, use_container_width=True, hide_index=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Aspek utama", result["aspect"])
        m2.metric("Sentimen", result["sentiment"])
        m3.metric("Confidence", result["confidence"])

        if probs is not None and id2label_sentiment is not None:
            prob_rows = []
            for idx, prob in enumerate(probs):
                prob_rows.append({
                    "Label": id2label_sentiment.get(int(idx), str(idx)),
                    "Probabilitas": round(float(prob), 4),
                })
            st.subheader("Probabilitas Sentimen Model")
            st.dataframe(pd.DataFrame(prob_rows), use_container_width=True, hide_index=True)

        with st.expander("Token label dari model ekstraksi aspek"):
            if token_labels:
                st.dataframe(
                    pd.DataFrame(token_labels, columns=["Token", "Label BIO"]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.write("Tidak ada token label.")

        if model_loaded:
            st.success("Model berhasil dijalankan.")
        else:
            st.warning("Model belum berhasil dijalankan. Cek error di atas dan folder model.")
