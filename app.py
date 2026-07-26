"""Streamlit app — Klasifikasi Amar Putusan Pidana IndoBERT (Clean Form Wizard)."""

import json
import os
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, BertForSequenceClassification

# --------------------------------------------------------------------------
# Konfigurasi
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR
MODEL_STATE_PATH = os.path.join(MODEL_DIR, "model_state.pt")
CONFIG_PATH = os.path.join(MODEL_DIR, "config_inferensi.json")
TOKENIZER_DIR = os.path.join(MODEL_DIR, "tokenizer")

HF_REPO_ID = "Scabiosa/indobert-law-classifier"

MODEL_NAME = "indolem/indobert-base-uncased"
MAX_LEN = 512
HEAD_LEN = 128
TAIL_LEN = 382

LABEL_MAP = {0: "pidana", 1: "bebas", 2: "lepas"}
LABEL_DESC = {
    "pidana": "Terdakwa dinyatakan TERBUKTI bersalah dan dijatuhi hukuman pidana.",
    "bebas": "Terdakwa TIDAK TERBUKTI secara sah dan meyakinkan — dibebaskan dari segala dakwaan.",
    "lepas": "Perbuatan TERBUKTI, namun BUKAN merupakan tindak pidana — terdakwa dilepaskan.",
}
LABEL_COLOR = {
    "pidana": "#B3261E",
    "bebas": "#1B7A4C",
    "lepas": "#B5720A",
}

CONTOH_TEKS = (
    "Menimbang bahwa terdakwa didakwa oleh Penuntut Umum dengan dakwaan melanggar "
    "Pasal 362 KUHP tentang pencurian. Berdasarkan keterangan saksi-saksi dan alat "
    "bukti yang diajukan di persidangan, terungkap fakta bahwa terdakwa terbukti "
    "mengambil barang milik korban tanpa izin pada malam hari dengan cara memanjat "
    "pagar rumah korban. Terdakwa mengakui perbuatannya di persidangan dan tidak ada "
    "alasan pemaaf maupun alasan pembenar yang dapat menghapuskan pertanggungjawaban "
    "pidana terdakwa. Majelis Hakim berpendapat bahwa seluruh unsur pasal yang "
    "didakwakan telah terpenuhi secara sah dan meyakinkan menurut hukum."
)

# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------


def clean_text(teks: str) -> str:
    teks = str(teks).lower()
    teks = re.sub(r"<[^>]+>", " ", teks)
    teks = re.sub(r"&[a-z]+;", " ", teks)
    teks = re.sub(r"[;:\(\)\[\]\{\}\/\\\"'`]", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks


def truncate_head_tail(teks: str, tokenizer):
    tokens = tokenizer.encode(teks, add_special_tokens=False)
    if len(tokens) <= 510:
        return teks, len(tokens), False
    head = tokens[:HEAD_LEN]
    tail = tokens[-TAIL_LEN:]
    truncated = tokenizer.decode(head + tail, skip_special_tokens=True)
    return truncated, len(tokens), True


# --------------------------------------------------------------------------
# Load model & konfigurasi
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_model_state_path() -> str:
    if os.path.exists(MODEL_STATE_PATH):
        return MODEL_STATE_PATH
    try:
        return hf_hub_download(repo_id=HF_REPO_ID, filename="model_state.pt")
    except Exception as e:
        raise FileNotFoundError(
            f"File model tidak ditemukan di '{MODEL_STATE_PATH}' dan gagal "
            f"diunduh dari Hugging Face Hub ('{HF_REPO_ID}'): {e}"
        )


def _resolve_tokenizer():
    if os.path.isdir(TOKENIZER_DIR):
        return AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    try:
        return AutoTokenizer.from_pretrained(HF_REPO_ID, subfolder="tokenizer")
    except Exception as e:
        raise FileNotFoundError(
            f"Folder tokenizer tidak ditemukan di '{TOKENIZER_DIR}' dan gagal "
            f"diunduh dari Hugging Face Hub ('{HF_REPO_ID}'): {e}"
        )


@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = _resolve_tokenizer()
    model_state_path = _resolve_model_state_path()

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True
    )
    try:
        state = torch.load(model_state_path, map_location=device)
    except RuntimeError:
        # fallback jika terjadi masalah memori GPU saat load
        device = torch.device("cpu")
        state = torch.load(model_state_path, map_location=device)

    model.load_state_dict(state)
    model = model.to(device).eval()

    return model, tokenizer, device


# --------------------------------------------------------------------------
# Prediksi
# --------------------------------------------------------------------------


def predict(teks_input: str, model, tokenizer, device):
    teks_clean = clean_text(teks_input)
    teks_trunc, n_tokens_asli, dipotong = truncate_head_tail(teks_clean, tokenizer)

    enc = tokenizer(
        teks_trunc,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    try:
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            probs = F.softmax(out.logits, dim=-1)[0].cpu().numpy()
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device.type == "cuda":
            torch.cuda.empty_cache()
            model_cpu = model.to("cpu")
            with torch.no_grad():
                out = model_cpu(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )
                probs = F.softmax(out.logits, dim=-1)[0].numpy()
        else:
            raise

    pred_idx = int(probs.argmax())
    pred_label = LABEL_MAP[pred_idx]
    confidence = float(probs.max())

    n_tokens_final = int(enc["attention_mask"].sum().item())

    return {
        "label": pred_label,
        "confidence": confidence,
        "probs": {
            "pidana": float(probs[0]),
            "bebas": float(probs[1]),
            "lepas": float(probs[2]),
        },
        "deskripsi": LABEL_DESC[pred_label],
        "color": LABEL_COLOR[pred_label],
        "teks_clean": teks_clean,
        "teks_trunc": teks_trunc,
        "n_tokens_asli": n_tokens_asli,
        "n_tokens_final": n_tokens_final,
        "dipotong": dipotong,
    }


# --------------------------------------------------------------------------
# Page config & CSS
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Klasifikasi Amar Putusan Pidana | IndoBERT",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --primary: #2A4365;
        --primary-light: #3A5A80;
        --success: #1B7A4C;
        --danger:  #B3261E;
        --warning: #B5720A;
        --text-dark: #1A202C;
        --text-muted: #4A5568;
        --bg-card: #F7F8FA;
        --border:  #D8DEE6;
    }

    html, body, [class*="css"] {
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }

    h1, h2, h3 {
        color: var(--text-dark);
        font-weight: 700;
    }

    /* ---- Step indicator ---- */
    .step-indicator {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 6px;
        margin-bottom: 24px;
    }
    .step-dot {
        width: 32px; height: 32px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700;
        font-size: 14px;
        border: 1px solid transparent;
    }
    .step-dot.active   { background: var(--primary); color: #FFFFFF; }
    .step-dot.done     { background: var(--success); color: #FFFFFF; }
    .step-dot.inactive { background: #E9ECEF; color: #6C757D; border-color: var(--border); }
    .step-line {
        width: 44px; height: 2px;
        background: var(--border);
    }
    .step-label {
        text-align: center;
        color: var(--text-muted);
        font-size: 13px;
        margin-bottom: 6px;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }

    /* ---- Info box ---- */
    .info-box {
        background: #EDF2FA;
        border: 1px solid #C4D3E8;
        border-left: 4px solid var(--primary);
        border-radius: 6px;
        padding: 14px 18px;
        font-size: 14px;
        line-height: 1.55;
        margin-bottom: 16px;
        color: var(--text-dark) !important;
    }
    .info-box, .info-box * {
        color: var(--text-dark) !important;
    }

    /* ---- Result card ---- */
    .result-card {
        padding: 26px 28px;
        border-radius: 8px;
        border: 1px solid var(--border);
        border-left: 5px solid;
        margin-bottom: 16px;
        background: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .result-label-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
    }
    .result-dot {
        width: 12px; height: 12px;
        border-radius: 50%;
        display: inline-block;
    }
    .result-title {
        font-size: 24px;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .result-confidence {
        font-size: 14px;
        color: var(--text-muted);
        margin-bottom: 12px;
    }
    .result-desc {
        font-size: 14.5px;
        color: var(--text-dark);
        line-height: 1.6;
    }

    .confidence-label {
        font-size: 13px;
        color: var(--text-muted);
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: var(--text-dark);
    }
    .sidebar-section-title {
        font-size: 12px;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 4px;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

if "step" not in st.session_state:
    st.session_state["step"] = 1
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""
if "result" not in st.session_state:
    st.session_state["result"] = None
if "input_mode" not in st.session_state:
    st.session_state["input_mode"] = "Ketik / Paste Teks"


def reset_wizard():
    st.session_state["step"] = 1
    st.session_state["input_text"] = ""
    st.session_state["text_area_input"] = ""
    st.session_state["result"] = None


def render_step_indicator(current_step: int):
    labels = ["Input Dokumen", "Proses Analisis", "Hasil Prediksi"]
    dots_html = '<div class="step-indicator">'
    for i in range(1, 4):
        if i < current_step:
            cls = "done"
            content = "&#10003;"
        elif i == current_step:
            cls = "active"
            content = str(i)
        else:
            cls = "inactive"
            content = str(i)
        dots_html += f'<div class="step-dot {cls}">{content}</div>'
        if i < 3:
            dots_html += '<div class="step-line"></div>'
    dots_html += "</div>"
    st.markdown(dots_html, unsafe_allow_html=True)
    st.markdown(
        f'<div class="step-label">Langkah {current_step} dari 3 — {labels[current_step - 1]}</div>',
        unsafe_allow_html=True,
    )
    st.progress(current_step / 3)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Tentang Proyek")
    st.markdown(
        """
        Aplikasi ini merupakan sistem klasifikasi otomatis untuk memprediksi
        **amar putusan** dalam perkara pidana di Indonesia, berdasarkan teks
        dokumen putusan pengadilan. Model membaca bagian dakwaan, fakta
        persidangan, dan pertimbangan hukum, kemudian mengklasifikasikan
        dokumen ke dalam salah satu dari tiga kategori:

        - **Pidana** — terdakwa terbukti bersalah dan dijatuhi hukuman.
        - **Bebas** — terdakwa tidak terbukti bersalah secara sah dan
          meyakinkan.
        - **Lepas** — perbuatan terbukti namun bukan merupakan tindak
          pidana, sehingga terdakwa dilepaskan dari segala tuntutan hukum.

        Model dibangun di atas arsitektur **IndoBERT**
        (`indolem/indobert-base-uncased`) yang di-*fine-tune* khusus untuk
        tugas ini menggunakan dataset **Indo-Law**.
        """
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">Ringkasan Model</div>', unsafe_allow_html=True)
    st.markdown(
        """
        | Parameter | Nilai |
        |---|---|
        | Basis model | IndoBERT (indolem/indobert-base-uncased) |
        | Dataset pelatihan | Indo-Law (22.462 dokumen) |
        | Akurasi | 99.72% |
        | F1 Macro | 0.9007 |
        """
    )

    st.markdown('<div class="sidebar-section-title">Performa per Kelas</div>', unsafe_allow_html=True)
    st.markdown(
        """
        | Kelas | F1-Score |
        |---|---|
        | Pidana | 0.9955 |
        | Bebas | 0.7671 |
        | Lepas | 0.9375 |
        """
    )

    st.markdown("---")

    with st.expander("Cara Menggunakan Aplikasi", expanded=False):
        st.markdown(
            """
            **1. Siapkan teks dokumen**
            Salin bagian dakwaan, fakta persidangan, dan pertimbangan
            hukum dari putusan yang ingin dianalisis. Anda bisa mengetik
            atau menempelkan teks langsung, atau mengunggah berkas
            `.txt`. Panjang teks minimal sekitar 10 kata.

            **2. Jalankan analisis**
            Klik tombol *Analisis Sekarang* untuk memulai proses. Sistem
            akan membersihkan teks, memotong bagian yang melebihi batas
            token model (menyisakan bagian awal dan akhir dokumen yang
            paling informatif), lalu menjalankan model klasifikasi.

            **3. Baca hasil prediksi**
            Hasil akan menampilkan label prediksi (Pidana, Bebas, atau
            Lepas), tingkat keyakinan model (*confidence*), dan
            distribusi probabilitas untuk ketiga kelas. Jika tingkat
            keyakinan di bawah 70%, sistem akan menyarankan verifikasi
            manual oleh praktisi hukum.

            **4. Ulangi untuk dokumen lain**
            Gunakan tombol *Analisis Dokumen Lain* untuk mengosongkan
            input dan memulai analisis baru.
            """
        )

    with st.expander("Disclaimer", expanded=False):
        st.markdown(
            """
            Sistem ini dikembangkan untuk **tujuan referensi dan riset
            akademis** semata. Hasil klasifikasi merupakan estimasi
            statistik dari model machine learning dan **bukan pengganti**
            putusan resmi pengadilan maupun pendapat hukum profesional.
            Keputusan hukum tetap harus diverifikasi dan divalidasi oleh
            praktisi hukum yang berwenang.
            """
        )

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("Klasifikasi Amar Putusan Pidana — IndoBERT")
render_step_indicator(st.session_state["step"])

# --------------------------------------------------------------------------
# STEP 1 — Input Dokumen
# --------------------------------------------------------------------------

if st.session_state["step"] == 1:
    st.header("Masukkan Teks Dokumen Hukum")

    st.markdown(
        """
        <div class="info-box">
        Sistem akan menganalisis bagian dakwaan, fakta, dan pertimbangan
        hukum dari teks yang dimasukkan.
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Pilih mode input:",
        ["Ketik / Paste Teks", "Upload File .txt"],
        horizontal=True,
        key="input_mode",
    )

    if st.button("Coba dengan Contoh Teks"):
        st.session_state["input_text"] = CONTOH_TEKS
        st.session_state["text_area_input"] = CONTOH_TEKS
        st.rerun()

    if mode == "Ketik / Paste Teks":
        if "text_area_input" not in st.session_state:
            st.session_state["text_area_input"] = st.session_state["input_text"]
        current_text = st.text_area(
            "Teks putusan:",
            height=200,
            placeholder="Tempel atau ketik teks amar putusan di sini...",
            key="text_area_input",
        )
        st.session_state["input_text"] = current_text
    else:
        current_text = st.session_state["input_text"]
        uploaded = st.file_uploader("Unggah file .txt", type=["txt"])
        if uploaded is not None:
            try:
                current_text = uploaded.read().decode("utf-8")
            except UnicodeDecodeError:
                uploaded.seek(0)
                current_text = uploaded.read().decode("latin-1")
            st.session_state["input_text"] = current_text
            st.text_area("Pratinjau teks:", value=current_text, height=200, disabled=True)

    n_char = len(current_text)
    n_word = len(current_text.split())
    st.caption(f"{n_char} karakter | {n_word} kata (estimasi)")

    col1, col2 = st.columns([3, 1])
    with col1:
        disabled = n_word < 10
        if disabled and n_char > 0:
            st.warning("Teks terlalu pendek. Minimal 10 kata diperlukan untuk analisis.")
        if st.button("Analisis Sekarang", type="primary", disabled=disabled or n_char == 0):
            st.session_state["step"] = 2
            st.rerun()
    with col2:
        if st.button("Reset"):
            reset_wizard()
            st.rerun()

# --------------------------------------------------------------------------
# STEP 2 — Proses Analisis
# --------------------------------------------------------------------------

elif st.session_state["step"] == 2:
    st.header("Memproses Dokumen")

    teks_input = st.session_state["input_text"]

    with st.spinner("Memuat model IndoBERT (mungkin memerlukan beberapa detik pada percobaan pertama)..."):
        try:
            model, tokenizer, device = load_model()
        except FileNotFoundError as e:
            st.error(f"Gagal memuat model: {e}")
            if st.button("Kembali ke Step 1"):
                st.session_state["step"] = 1
                st.rerun()
            st.stop()

    n_char_asli = len(teks_input)
    n_word_asli = len(teks_input.split())
    teks_clean = clean_text(teks_input)

    with st.spinner("Menganalisis teks dengan model..."):
        try:
            result = predict(teks_input, model, tokenizer, device)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")
            if st.button("Kembali ke Step 1"):
                st.session_state["step"] = 1
                st.rerun()
            st.stop()

    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown("**Panjang teks asli**")
        st.write(f"{n_char_asli} karakter | {n_word_asli} kata")
        st.markdown("**Panjang setelah cleaning**")
        st.write(f"{len(teks_clean)} karakter")
    with info_col2:
        st.markdown("**Strategi truncation**")
        if result["dipotong"]:
            st.write(f"Head+Tail ({HEAD_LEN} awal + {TAIL_LEN} akhir token)")
        else:
            st.write("Tidak diperlukan (teks di bawah batas 510 token)")
        st.markdown("**Jumlah token hasil tokenisasi**")
        st.write(f"{result['n_tokens_asli']} token asli → {result['n_tokens_final']} token final")

    st.session_state["result"] = result
    st.session_state["step"] = 3
    st.rerun()

# --------------------------------------------------------------------------
# STEP 3 — Hasil Prediksi
# --------------------------------------------------------------------------

elif st.session_state["step"] == 3:
    st.header("Hasil Analisis Amar Putusan")

    result = st.session_state["result"]

    if result is None:
        st.warning("Belum ada hasil analisis. Silakan mulai dari Step 1.")
        if st.button("Ke Step 1"):
            reset_wizard()
            st.rerun()
        st.stop()

    col_left, col_right = st.columns([6, 4])

    with col_left:
        label = result["label"]
        conf = result["confidence"]
        color = result["color"]

        st.markdown(
            f"""
            <div class="result-card" style="border-left-color: {color};">
                <div class="result-label-row">
                    <span class="result-dot" style="background: {color};"></span>
                    <span class="result-title" style="color: {color};">{label.upper()}</span>
                </div>
                <div class="result-confidence">Confidence: {conf * 100:.1f}%</div>
                <hr style="border-color: var(--border); margin: 10px 0;">
                <div class="result-desc">{result['deskripsi']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="confidence-label">Tingkat Keyakinan Model</div>', unsafe_allow_html=True)
        st.progress(conf)

        if conf < 0.70:
            st.warning(
                "Confidence rendah. Disarankan verifikasi manual oleh praktisi hukum."
            )

    with col_right:
        st.subheader("Distribusi Probabilitas")

        probs = result["probs"]
        classes = list(probs.keys())
        values = [probs[c] * 100 for c in classes]
        colors = [LABEL_COLOR[c] for c in classes]

        fig = go.Figure(
            go.Bar(
                x=values,
                y=classes,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.2f}%" for v in values],
                textposition="auto",
            )
        )
        fig.update_layout(
            xaxis_title="Probabilitas (%)",
            yaxis_title=None,
            margin=dict(l=0, r=10, t=10, b=30),
            height=220,
            xaxis=dict(range=[0, 100]),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.table(
            {
                "Kelas": classes,
                "Probabilitas": [f"{probs[c] * 100:.2f}%" for c in classes],
            }
        )

    st.markdown("---")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Analisis Dokumen Lain", type="primary"):
            reset_wizard()
            st.rerun()
    with btn_col2:
        summary = (
            f"Hasil Klasifikasi Amar Putusan: {result['label'].upper()}\n"
            f"Confidence: {result['confidence'] * 100:.1f}%\n"
            f"Deskripsi: {result['deskripsi']}\n"
            f"Probabilitas — pidana: {result['probs']['pidana'] * 100:.2f}%, "
            f"bebas: {result['probs']['bebas'] * 100:.2f}%, "
            f"lepas: {result['probs']['lepas'] * 100:.2f}%"
        )
        if st.button("Salin Hasil"):
            st.code(summary, language=None)
            st.info("Salin teks di atas untuk menyalin hasil analisis.")
