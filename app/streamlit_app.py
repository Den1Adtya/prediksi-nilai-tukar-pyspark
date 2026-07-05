# =========================================================
# DASHBOARD STREAMLIT - PREDIKSI NILAI TUKAR USD/IDR
# =========================================================
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import time
import random
import os
import json
from datetime import datetime

# -------------------------------------------------------
# KONFIGURASI HALAMAN (harus jadi perintah st. pertama)
# -------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Nilai Tukar USD/IDR",
    page_icon="💱",
    layout="wide"
)

# -------------------------------------------------------
# PATH & KONSTANTA
# -------------------------------------------------------
CSV_PATH = "forex_ml_project/data/historical_rates.csv"
MODEL_PATH = "forex_ml_project/models/linear_regression_model"
FEATURE_CONFIG_PATH = "forex_ml_project/models/feature_config.json"


# -------------------------------------------------------
# CACHING: memuat Spark session & model HANYA SEKALI
# -------------------------------------------------------
# @st.cache_resource memastikan objek berat (Spark session, model) tidak
# dibuat ulang setiap kali dashboard di-refresh/rerun — hanya dibuat sekali
# lalu disimpan di memori server Streamlit selama aplikasi berjalan.
@st.cache_resource
def load_spark_and_model():
    from pyspark.sql import SparkSession
    from pyspark.ml.regression import LinearRegressionModel

    spark_session = SparkSession.builder \
        .appName("StreamlitForexApp") \
        .master("local[*]") \
        .getOrCreate()

    model = LinearRegressionModel.load(MODEL_PATH)

    with open(FEATURE_CONFIG_PATH, "r") as f:
        config = json.load(f)

    return spark_session, model, config["feature_cols"]


spark, lr_model, feature_cols = load_spark_and_model()


# -------------------------------------------------------
# FUNGSI: ambil rate terbaru dari API (sama seperti Tahap 2)
# -------------------------------------------------------
def fetch_latest_rate():
    url = "https://api.frankfurter.app/latest"
    params = {"from": "USD", "to": "IDR"}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    return {"date": data["date"], "rate": data["rates"]["IDR"]}


# -------------------------------------------------------
# FUNGSI: hitung fitur & prediksi (logika sama seperti Tahap 8)
# -------------------------------------------------------
def predict_next_rate(df_recent: pd.DataFrame):
    from pyspark.ml.feature import VectorAssembler

    df_recent = df_recent.copy()
    df_recent["lag_1"] = df_recent["rate"].shift(1)
    df_recent["lag_2"] = df_recent["rate"].shift(2)
    df_recent["lag_3"] = df_recent["rate"].shift(3)
    df_recent["ma_3"] = df_recent["rate"].rolling(3).mean()
    df_recent["ma_7"] = df_recent["rate"].rolling(7).mean()
    df_recent["rate_change"] = df_recent["rate"] - df_recent["lag_1"]
    df_recent["day_of_week"] = pd.to_datetime(df_recent["date"], errors="coerce").dt.dayofweek.fillna(0) + 1

    last_row = df_recent.iloc[[-1]][feature_cols]
    if last_row.isnull().any(axis=1).values[0]:
        return None

    spark_row = spark.createDataFrame(last_row)
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    vector_row = assembler.transform(spark_row)
    prediction = lr_model.transform(vector_row).collect()[0]["prediction"]
    return round(prediction, 2)


# -------------------------------------------------------
# SIDEBAR: kontrol pengguna
# -------------------------------------------------------
st.sidebar.title("⚙️ Kontrol Dashboard")
auto_refresh = st.sidebar.checkbox("Aktifkan auto-refresh (simulasi live)", value=False)
refresh_interval = st.sidebar.slider("Interval refresh (detik)", 5, 30, 7)
demo_mode = st.sidebar.checkbox("Mode Demo (tambah variasi kecil pada data)", value=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Sumber data: Frankfurter API (ECB reference rates). "
    "Mode Demo menambahkan variasi kecil (±0.15%) agar pergerakan terlihat "
    "saat presentasi, karena data ECB hanya update sekali per hari kerja."
)

# -------------------------------------------------------
# HEADER UTAMA
# -------------------------------------------------------
st.title("💱 Dashboard Prediksi Nilai Tukar USD → IDR")
st.caption("Dibangun dengan PySpark, Spark MLlib, dan Streamlit")

# -------------------------------------------------------
# AMBIL DATA TERBARU & LAKUKAN PREDIKSI
# -------------------------------------------------------
df_hist = pd.read_csv(CSV_PATH)

with st.spinner("Mengambil data terbaru dari API..."):
    latest = fetch_latest_rate()

if demo_mode:
    noise = 1 + random.uniform(-0.0015, 0.0015)
    latest["rate"] = round(latest["rate"] * noise, 2)

df_for_prediction = pd.concat([df_hist, pd.DataFrame([latest])]).tail(10).reset_index(drop=True)
prediction = predict_next_rate(df_for_prediction)

# -------------------------------------------------------
# METRIC CARDS (ringkasan angka penting di bagian atas)
# -------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Rate Saat Ini (USD → IDR)", f"Rp {latest['rate']:,.2f}")

if prediction is not None:
    selisih = prediction - latest["rate"]
    col2.metric("Prediksi Rate Berikutnya", f"Rp {prediction:,.2f}", delta=f"{selisih:,.2f}")
else:
    col2.metric("Prediksi Rate Berikutnya", "Data belum cukup")

col3.metric("Update Terakhir", datetime.now().strftime("%H:%M:%S"))

# -------------------------------------------------------
# GRAFIK HISTORIS
# -------------------------------------------------------
st.subheader("📈 Grafik Historis Nilai Tukar")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=pd.to_datetime(df_hist["date"]), y=df_hist["rate"],
    mode="lines", name="Rate Historis", line=dict(color="royalblue")
))
fig.add_trace(go.Scatter(
    x=[pd.to_datetime(latest["date"])], y=[latest["rate"]],
    mode="markers", name="Rate Terbaru",
    marker=dict(color="orange", size=12, symbol="star")
))
fig.update_layout(template="plotly_dark", xaxis_title="Tanggal", yaxis_title="IDR", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------
# TABEL DATA TERBARU
# -------------------------------------------------------
st.subheader("🗂️ Data Historis (10 Terakhir)")
st.dataframe(df_hist.tail(10), use_container_width=True)

# -------------------------------------------------------
# AUTO-REFRESH (untuk mensimulasikan "live" streaming)
# -------------------------------------------------------
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
