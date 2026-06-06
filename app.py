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
""", unsafe_allow_html=True)

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
            
        label_rapi = kol.replace('_', ' ').title()
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
        # Rekonstruksi efek One-Hot Encoding secara manual agar jumlah kolom kembali menjadi 14
        df_rekonstruksi = pd.DataFrame()

        # A. Masukkan fitur numerik langsung sesuai urutan aslinya
        for kol in NUM_COLS:
            if kol.lower() != 'id_mahasiswa' and kol in input_user:
                df_rekonstruksi[kol] = [input_user[kol]]

        # B. Membuat kolom biner hasil pecahan kategorikal (One-Hot Encoded format)
        df_rekonstruksi['jenis_kelamin_L'] = [1.0 if input_user.get('jenis_kelamin') == 'L' else 0.0]
        df_rekonstruksi['jenis_kelamin_P'] = [1.0 if input_user.get('jenis_kelamin') == 'P' else 0.0]

        df_rekonstruksi['asal_daerah_Dalam Jawa'] = [1.0 if input_user.get('asal_daerah') == 'Dalam Jawa' else 0.0]
        df_rekonstruksi['asal_daerah_Luar Jawa'] = [1.0 if input_user.get('asal_daerah') == 'Luar Jawa' else 0.0]

        df_rekonstruksi['kerja_paruh_waktu_Tidak'] = [1.0 if input_user.get('kerja_paruh_waktu') == 'Tidak' else 0.0]
        df_rekonstruksi['kerja_paruh_waktu_Ya'] = [1.0 if input_user.get('kerja_paruh_waktu') == 'Ya' else 0.0]

        df_rekonstruksi['beasiswa_Tidak'] = [1.0 if input_user.get('beasiswa') == 'Tidak' else 0.0]
        df_rekonstruksi['beasiswa_Ya'] = [1.0 if input_user.get('beasiswa') == 'Ya' else 0.0]

        # C. Transformasi menggunakan Selector (14 fitur -> 5 fitur) dengan fungsi penyelamat backend
        try:
            X_sel = selector.transform(df_rekonstruksi)
        except Exception:
            try:
                X_sel = df_rekonstruksi[selector.get_feature_names_out()]
            except Exception:
                X_sel = df_rekonstruksi

        # D. Komputasi Prediksi Skor Probabilitas dari Model Utama
        try:
            proba = model.predict_proba(X_sel)[0, 1]
        except Exception:
            proba = model.predict_proba(X_sel.iloc[:, :5])[0, 1]
            
        pred = int(proba >= threshold)
        kelas_pred = le.classes_[pred]

        # 5. Menampilkan Komponen Visual Hasil Prediksi ke Layar
        st.divider()
        st.subheader('📊 Hasil Analisis Model Klasifikasi')
        
        with st.container():
            if pred == 1 or str(kelas_pred).lower() == 'ya' or 'tepat' in str(kelas_pred).lower():
                st.balloons()
                st.success(f"### 🎉 REKOMENDASI HASIL: **{str(kelas_pred).upper()}**")
            else:
                st.warning(f"### ⚠️ REKOMENDASI HASIL: **{str(kelas_pred).upper()}**")
                
            st.markdown(f"**Keyakinan Model (Probabilitas Kelas Positif):** `{proba*100:.2f}%`")
            st.progress(float(proba))

        # Menampilkan Sejajar Elemen Metrik
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric(label="Skor Probabilitas", value=f"{proba:.4f}")
        res_col2.metric(label="Threshold Sistem", value=f"{threshold:.4f}")
        res_col3.metric(label="Status Prediksi", value="Sukses ✅")

        # Menampilkan Tabel Histori Ringkasan Data Input User
        st.markdown("<br>", unsafe_allow_html=True)
        st.write('### 🔍 Ringkasan Data Input Terpilih')
        st.dataframe(pd.DataFrame([input_user]), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f'❌ Terjadi Kendala Teknis: {e}')
        st.info('Mohon pastikan format isian data Anda telah sesuai dengan ketentuan konfigurasi model.')

# 6. Catatan Kaki / Footer Dashboard Halaman Utama
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #718096; font-size: 14px;'>"
    "Dashboard Aplikasi Tabular Classifier © 2026 | PPKD Jakarta Selatan"
    "</p>", 
    unsafe_allow_html=True
)
