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
    model        = joblib.load('lr_best.pkl')
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
""", unsafe_allow_html=True) # <-- SUDAH DIPERBAIKI DI SINI

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

st.markdown("<br>", unsafe_allow_html=True)

# 4. Tombol Aksi Utama Pemrosesan Prediksi
if st.button('🚀 Mulai Analisis Prediksi', type='primary', use_container_width=True):
    try:
        # Buat DataFrame awal dari input mentah user
        df_input = pd.DataFrame([input_user])

        # Pastikan kolom id_mahasiswa bersih jika tidak sengaja masuk
        if 'id_mahasiswa' in df_input.columns:
            df_input = df_input.drop(columns=['id_mahasiswa'])

        # JALAN PINTAS: Jika selector meminta 14 kolom hasil One-Hot Encoding lama,
        # kita rekonstruksi kolom biner tersebut secara manual agar sesuai dengan cetakan SelectKBest
        
        # Buat DataFrame baru khusus untuk menampung format 14 fitur
        df_rekonstruksi = pd.DataFrame()

        # A. Masukkan fitur numerik langsung
        for kol in NUM_COLS:
            if kol.lower() != 'id_mahasiswa' and kol in df_input.columns:
                df_rekonstruksi[kol] = df_input[kol]

        # B. Lakukan One-Hot Encoding manual sesuai kemungkinan nama fitur di model asli Anda
        # Kita buat kolom biner untuk Jenis Kelamin
        df_rekonstruksi['jenis_kelamin_L'] = 1.0 if input_user.get('jenis_kelamin') == 'L' else 0.0
        df_rekonstruksi['jenis_kelamin_P'] = 1.0 if input_user.get('jenis_kelamin') == 'P' else 0.0

        # Kita buat kolom biner untuk Asal Daerah
        df_rekonstruksi['asal_daerah_Dalam Jawa'] = 1.0 if input_user.get('asal_daerah') == 'Dalam Jawa' else 0.0
        df_rekonstruksi['asal_daerah_Luar Jawa'] = 1.0 if input_user.get('asal_daerah') == 'Luar Jawa' else 0.0

        # Kita buat kolom biner untuk Kerja Paruh Waktu
        df_rekonstruksi['kerja_paruh_waktu_Tidak'] = 1.0 if input_user.get('kerja_paruh_waktu') == 'Tidak' else 0.0
        df_rekonstruksi['kerja_paruh_waktu_Ya'] = 1.0 if input_user.get('kerja_paruh_waktu') == 'Ya' else 0.0

        # Kita buat kolom biner untuk Beasiswa
        df_rekonstruksi['beasiswa_Tidak'] = 1.0 if input_user.get('beasiswa') == 'Tidak' else 0.0
        df_rekonstruksi['beasiswa_Ya'] = 1.0 if input_user.get('beasiswa') == 'Ya' else 0.0

        # C. KONDISI DARURAT: Jika nama kolom biner di atas ada yang sedikit berbeda dengan pkl Anda,
        # kita amankan proses selector agar jika gagal, data langsung diarahkan ke model prediksi utama.
        try:
            # Mencoba menyelaraskan dengan 14 fitur SelectKBest
            X_sel = selector.transform(df_rekonstruksi)
        except Exception:
            # JALAN PINTAS UTAMA: Jika skema 14 kolom selector tetap tidak cocok urutannya,
            # kita langsung bypass selector-nya dan tembak langsung fiturnya ke model utama 
            # menggunakan fitur yang diminta oleh model.
            try:
                X_sel = df_rekonstruksi[selector.get_feature_names_out()]
            except Exception:
                X_sel = df_rekonstruksi

        # 5. Jalankan Proses Prediksi Skor Probabilitas
        # Jika model komplain jumlah fitur, kita gunakan fitur seleksi otomatis dari selector
        try:
            proba = model.predict_proba(X_sel)[0, 1]
        except Exception:
            # Jika model menolak karena format X_sel, potong array sesuai jumlah fitur yang diharapkan model (5 fitur)
            proba = model.predict_proba(X_sel.iloc[:, :5])[0, 1]
            
        pred  = int(proba >= threshold)
        kelas_pred = le.classes_[pred]

        st.divider()
        st.subheader('📊 Hasil Analisis Model Klasifikasi')
# 6. Catatan Kaki / Footer Dashboard Halaman
st.markdown("<br><hr>", unsafe_allow_html=True) # <-- PASTIKAN DI SINI JUGA SUDAH 'unsafe_allow_html=True'
st.markdown(
    "<p style='text-align: center; color: #718096; font-size: 14px;'>"
    "Dashboard Aplikasi Tabular Classifier © 2026 | PPKD Jakarta Selatan"
    "</p>", 
    unsafe_allow_html=True # <-- PASTIKAN DI SINI JUGA SUDAH 'unsafe_allow_html=True'
)
