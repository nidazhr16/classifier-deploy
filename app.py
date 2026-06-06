import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Konfigurasi Halaman Utama dengan Layout Wide dan Tema Bersih
st.set_page_config(
    page_title='Dashboard Prediksi Kelulusan Mahasiswa', 
    page_icon='🎓', 
    layout='wide'
)

@st.cache_resource
def load_artefak():
    # Memuat artefak model dan kelengkapannya dari repositori GitHub
    model        = joblib.load('model.pkl')
    selector     = joblib.load('selector.pkl')
    le           = joblib.load('label_encoder.pkl')
    meta         = joblib.load('meta.pkl')
    preprocessor = None  # Safe bypass

    # Memuat threshold dari file threshold.txt dengan pengaman
    try:
        with open('threshold.txt') as f:
            thr = float(f.read().strip())
    except Exception:
        thr = 0.590  # Nilai default jika terjadi kendala pembacaan file

    return model, preprocessor, selector, le, meta, thr

# Memanggil fungsi untuk memuat seluruh artefak
model, preprocessor, selector, le, meta, threshold = load_artefak()
NUM_COLS = meta['NUM_COLS']
CAT_COLS = meta['CAT_COLS']

# 2. Desain Header / Banner Atas Aplikasi
st.markdown("""
    <div style="background-color:#1E3A8A; padding:20px; border-radius:10px; margin-bottom:25px;">
        <h1 style="color:white; margin-top:0px; text-align:center;">🎓 Sistem Prediksi Kelulusan Mahasiswa</h1>
        <p style="color:#E2E8F0; font-size:16px; margin-bottom:0px; text-align:center;">
            Kejuruan Data Analyst — Pusat Pelatihan Kerja Daerah (PPKD) Jakarta Selatan
        </p>
    </div>
""", unsafe_index=True)

# Menampilkan informasi sistem dalam struktur kolom yang rapi
info_col1, info_col2 = st.columns(2)
with info_col1:
    st.info(f"🔍 **Sistem Menggunakan Batas Keputusan (Threshold):** `{threshold:.3f}`")
with info_col2:
    st.success(f"🏷️ **Target Kelas:** {', '.join([str(c) for c in le.classes_])}")

st.divider()

# 3. Form Input Fitur dengan Grid Columns
st.subheader('📋 Form Input Data Mahasiswa')
st.markdown('*Silakan lengkapi seluruh informasi di bawah ini untuk memulai proses prediksi:*')

col1, col2 = st.columns(2)
input_user = {}

with col1:
    st.write('### 🔢 Atribut Akademik (Numerik)')
    for kol in NUM_COLS:
        if kol.lower() == 'id_mahasiswa':
            continue
            
        # Memberikan label yang lebih rapi dan deskriptif untuk dibaca
        label_rapi = kol.replace('_', ' ').title()
        
        # Menyesuaikan nilai bawaan (default value) agar lebih logis untuk IPK
        val_default = 3.00 if 'ipk' in kol.lower() else 0.0
        
        input_user[kol] = st.number_input(
            label=f"➡️ {label_rapi}", value=val_default, step=0.01, format='%.4f', key=f'num_{kol}'
        )

with col2:
    st.write('### 👤 Profil & Latar Belakang (Kategorikal)')
    for kol in CAT_COLS:
        if kol.lower() == 'id_mahasiswa':
            continue
            
        label_rapi = kol.replace('_', ' ').title()
        
        if kol == 'jenis_kelamin':
            input_user[kol] = st.selectbox(label=f"➡️ {label_rapi}", options=['L', 'P'], key=f'cat_{kol}')
        elif kol == 'asal_daerah':
            input_user[kol] = st.selectbox(label=f"➡️ {label_rapi}", options=['Dalam Jawa', 'Luar Jawa'], key=f'cat_{kol}')
        elif kol in ['kerja_paruh_waktu', 'beasiswa']:
            input_user[kol] = st.selectbox(label=f"➡️ {label_rapi}", options=['Ya', 'Tidak'], key=f'cat_{kol}')
        else:
            input_user[kol] = st.text_input(label=f"➡️ {label_rapi}", value='', key=f'cat_{kol}')

st.markdown("<br>", unsafe_index=True)

# 4. Tombol Aksi Utama Pemrosesan Prediksi
if st.button('🚀 Mulai Analisis Prediksi', type='primary', use_container_width=True):
    try:
        # Proses duplikasi data untuk manipulasi angka biner
        data_proses = input_user.copy()

        # Encoding Kategorikal Otomatis (Mapping ke Angka Biner)
        if 'jenis_kelamin' in data_proses:
            data_proses['jenis_kelamin'] = 1 if data_proses['jenis_kelamin'] == 'L' else 0
        if 'asal_daerah' in data_proses:
            data_proses['asal_daerah'] = 1 if data_proses['asal_daerah'] == 'Luar Jawa' else 0
        if 'kerja_paruh_waktu' in data_proses:
            data_proses['kerja_paruh_waktu'] = 1 if data_proses['kerja_paruh_waktu'] == 'Ya' else 0
        if 'beasiswa' in data_proses:
            data_proses['beasiswa'] = 1 if data_proses['beasiswa'] == 'Ya' else 0

        df_input = pd.DataFrame([data_proses])

        if 'id_mahasiswa' in df_input.columns:
            df_input = df_input.drop(columns=['id_mahasiswa'])

        # Menghubungkan ke Selector Fitur (SelectKBest)
        try:
            X_sel = selector.transform(df_input)
        except Exception:
            X_sel = df_input

        # Komputasi Probabilitas Model
        proba = model.predict_proba(X_sel)[0, 1]
        pred  = int(proba >= threshold)
        kelas_pred = le.classes_[pred]

        st.divider()
        st.subheader('📊 Hasil Analisis Model Klasifikasi')
        
        # Desain visual kartu penanda hasil klasifikasi kelulusan
        with st.container():
            # Jika hasil prediksi bernilai positif/Lulus Tepat Waktu (disesuaikan dengan label target Anda)
            if pred == 1 or str(kelas_pred).lower() == 'ya' or 'tepat' in str(kelas_pred).lower():
                st.balloons()
                st.success(f"### 🎉 REKOMENDASI HASIL: **{str(kelas_pred).upper()}**")
            else:
                st.warning(f"### ⚠️ REKOMENDASI HASIL: **{str(kelas_pred).upper()}**")
                
            # Menampilkan Visualisasi Indikator Progress Bar Probabilitas
            st.markdown(f"**Keyakinan Model (Probabilitas Kelas Positif):** `{proba*100:.2f}%`")
            st.progress(float(proba))

        # Menampilkan Metrik Komparasi secara Sejajar (Card Metric)
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric(label="Skor Probabilitas", value=f"{proba:.4f}", delta="Akurasi Model")
        res_col2.metric(label="Threshold Sistem", value=f"{threshold:.4f}")
        res_col3.metric(label="Status Prediksi", value="Sukses ✅")

        # 5. Tabel Histori Input Data yang Dimasukkan
        st.markdown("<br>", unsafe_index=True)
        st.write('### 🔍 Ringkasan Data Input Terpilih')
        st.dataframe(pd.DataFrame([input_user]), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f'❌ Terjadi Kendala Teknis: {e}')
        st.info('Mohon pastikan format isian data Anda telah sesuai dengan ketentuan konfigurasi model.')

# 6. Catatan Kaki / Footer Dashboard Halaman
st.markdown("<br><hr>", unsafe_index=True)
st.markdown(
    "<p style='text-align: center; color: #718096; font-size: 14px;'>"
    "Dashboard Aplikasi Tabular Classifier © 2026 | PPKD Jakarta Selatan"
    "</p>", 
    unsafe_index=True
)
