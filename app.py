import streamlit as st
import joblib
import pandas as pd
import numpy as np
import category_encoders as ce

st.set_page_config(page_title='Classifier Tabular', page_icon=':bar_chart:', layout='wide')
 
@st.cache_resource
def load_artefak():
    model        = joblib.load('lr_best.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    selector     = joblib.load('selector.pkl')
    le           = joblib.load('label_encoder.pkl')
    meta         = joblib.load('meta.pkl')
    with open('threshold.txt') as f:
        thr = float(f.read().strip())
    return model, preprocessor, selector, le, meta, thr
 
model, preprocessor, selector, le, meta, threshold = load_artefak()
NUM_COLS = meta['NUM_COLS']
CAT_COLS = meta['CAT_COLS']
 
st.title(':bar_chart: Web Klasifikasi Tabular')
st.caption(f'Threshold prediksi: {threshold:.3f}  |  Kelas: {list(le.classes_)}')
st.divider()
 
# Form input fitur
st.subheader('Masukkan nilai fitur:')
col1, col2 = st.columns(2)
input_user = {}
 
with col1:
    st.markdown('**Fitur Numerik**')
    for kol in NUM_COLS:
        input_user[kol] = st.number_input(
            label=kol, value=0.0, step=0.1, format='%.4f', key=f'num_{kol}'
        )
 
with col2:
    st.markdown('**Fitur Kategorikal**')
    for kol in CAT_COLS:
        # Default: text input (peserta bisa ganti ke selectbox kalau tahu nilai uniknya)
        input_user[kol] = st.text_input(label=kol, value='', key=f'cat_{kol}')
 
st.divider()
if st.button('Prediksi', type='primary', use_container_width=True):
    try:
        # Susun DataFrame sesuai urutan fitur
        df_input = pd.DataFrame([input_user])
 
        # Apply preprocessing pipeline
        X_enc = preprocessor.transform(df_input)
        X_sel = selector.transform(X_enc)
 
        # Prediksi
        proba = model.predict_proba(X_sel)[0, 1]
        pred  = int(proba >= threshold)
        kelas_pred = le.classes_[pred]
 
        # Tampilkan hasil
        st.success(f'Hasil prediksi: **{kelas_pred}**')
 
        cm1, cm2 = st.columns(2)
        cm1.metric('Probabilitas kelas positif', f'{proba:.4f}')
        cm2.metric('Threshold yang dipakai',     f'{threshold:.4f}')
        st.progress(float(proba))
 
        # Tampilkan input yang dipakai
        st.subheader('Input yang Digunakan')
        st.dataframe(df_input, use_container_width=True, hide_index=True)
 
    except Exception as e:
        st.error(f'Error: {e}')
        st.info('Periksa nilai input - pastikan kategorikal sesuai dengan training data.')
 
st.divider()
st.caption('Dibuat untuk PPKD Jakarta Selatan - Kejuruan Data Analyst')
