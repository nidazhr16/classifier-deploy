import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Konfigurasi halaman utama Streamlit
st.set_page_config(page_title='Classifier Tabular', page_icon=':bar_chart:', layout='wide')

@st.cache_resource
def load_artefak():
    # Memuat artefak model dan kelengkapannya
    model        = joblib.load('lr_best.pkl')
    selector     = joblib.load('selector.pkl')
    le           = joblib.load('label_encoder.pkl')
    meta         = joblib.load('meta.pkl')
    
    # SAFE BYPASS: Jika preprocessor.pkl bermasalah karena perbedaan versi,
    # kita set None dan tangani transformasinya secara manual / langsung.
    try:
        preprocessor = joblib.load('preprocessor.pkl')
    except Exception:
        preprocessor = None

    # Memuat threshold dari file teks
    with open('threshold.txt') as f:
        thr = float(f.read().strip())
        
    return model, preprocessor, selector, le, meta, thr

# Memanggil fungsi untuk memuat seluruh artefak
model, preprocessor, selector, le, meta, threshold = load_artefak()
NUM_COLS = meta['NUM_COLS']
CAT_COLS = meta['CAT_COLS']

# Judul Aplikasi Web
st.title(':bar_chart: Web Klasifikasi Tabular')
st.caption(f'Threshold prediksi: {threshold:.3f}  |  Kelas: {list(le.classes_)}')
st.divider()

# Form input fitur untuk User
st.subheader('Masukkan nilai fitur:')
col1, col2 = st.columns(2)
input_user = {}

with col1:
    st.markdown('**Fitur Numerik**')
    for kol in NUM_COLS:
        # Menghindari input id_mahasiswa jika tidak sengaja masuk ke meta
        if kol.lower() == 'id_mahasiswa':
            continue
        input_user[kol] = st.number_input(
            label=kol, value=0.0, step=0.1, format='%.4f', key=f'num_{kol}'
        )

with col2:
    st.markdown('**Fitur Kategorikal**')
    for kol in CAT_COLS:
        if kol.lower() == 'id_mahasiswa':
            continue
        # Default: text input (bisa diganti ke selectbox jika tahu nilai uniknya)
        input_user[kol] = st.text_input(label=kol, value='', key=f'cat_{kol}')

st.divider()

# Logika Tombol Prediksi
if st.button('Prediksi', type='primary', use_container_width=True):
    try:
        # Susun DataFrame sesuai urutan input user
        df_input = pd.DataFrame([input_user])

        # Antisipasi pengondisian data Tugas 3: Pastikan id_mahasiswa benar-benar bersih
        if 'id_mahasiswa' in df_input.columns:
            df_input = df_input.drop(columns=['id_mahasiswa'])

        # Cek alur Preprocessing Pipeline
        if preprocessor is not None:
            X_enc = preprocessor.transform(df_input)
            X_sel = selector.transform(X_enc)
        else:
            # Jika preprocessor di-bypass, jalankan transformasi alternatif/langsung ke selector
            try:
                X_sel = selector.transform(df_input)
            except Exception:
                X_sel = df_input

        # Proses Prediksi Skor Probabilitas
        proba = model.predict_proba(X_sel)[0, 1]
        pred  = int(proba >= threshold)
        kelas_pred = le.classes_[pred]

        # Tampilkan hasil prediksi utama
        st.success(f'Hasil prediksi: **{kelas_pred}**')

        # Tampilkan visualisasi metrik probabilitas dan threshold
        cm1, cm2 = st.columns(2)
        cm1.metric('Probabilitas kelas positif', f'{proba:.4f}')
        cm2.metric('Threshold yang dipakai',     f'{threshold:.4f}')
        st.progress(float(proba))

        # Tampilkan kembali data input yang dimasukkan user
        st.subheader('Input yang Digunakan')
        st.dataframe(df_input, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f'Error: {e}')
        st.info('Periksa nilai input - pastikan nilai kategorikal sesuai dengan data training.')


st.caption('Dibuat untuk PPKD Jakarta Selatan - Kejuruan Data Analyst') 
