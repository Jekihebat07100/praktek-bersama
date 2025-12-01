import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model")

st.title("belajar machinelearning")
st.markdown("aplikasi untuk memprediksi siswa apakah diterima atau tidak")

Usia = st.slider("usia", 15.00000, 18.00000, 16.55000)
Nilai_Rerata = st.slider("nilai rerata", 60.200000, 99.620000, 80.273233)
Aktivitas_Ekstrakurikuler = st.slider("Aktivitas_Ekstrakurikuler", 0.000000, 5.000000, 2.486667)
Jenis_Kelamin = st.pills("jenis kelamin", ["L", "P"], default=["P"])

if st.button("prediksi", type="primary"):
	data_baru = pd.DataFrame([[15,65.8,5,"P"]],
                         columns=['Usia', 'Nilai_Rerata', 'Aktivitas_Ekstrakurikuler', 'Jenis_Kelamin'])
	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f"model memprediksi {prediksi} dengan tingkat keyakinan {presentase*100:.2f}%")