import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="💎 Diamond Dynamics",
    page_icon="💎",
    layout="centered"
)

# ── Load Models ───────────────────────────────────────────
@st.cache_resource
def load_models():
    reg_model     = joblib.load('../models/best_regression_model.pkl')
    cluster_model = joblib.load('../models/kmeans_model.pkl')
    scaler        = joblib.load('../models/scaler.pkl')
    scaler_cluster= joblib.load('../models/scaler_cluster.pkl')
    cluster_names = joblib.load('../models/cluster_names.pkl')
    return reg_model, cluster_model, scaler, scaler_cluster, cluster_names

reg_model, cluster_model, scaler, scaler_cluster, cluster_names = load_models()

# ── Helper: Feature Engineering ───────────────────────────
def prepare_input(carat, cut, color, clarity, depth, table, x, y, z):
    cut_map     = {'Fair':0,'Good':1,'Very Good':2,'Premium':3,'Ideal':4}
    color_map   = {'J':0,'I':1,'H':2,'G':3,'F':4,'E':5,'D':6}
    clarity_map = {'I1':0,'SI2':1,'SI1':2,'VS2':3,'VS1':4,'VVS2':5,'VVS1':6,'IF':7}

    volume          = x * y * z
    dimension_ratio = (x + y) / (2 * z)
    log_carat       = np.log1p(carat)
    cut_enc         = cut_map[cut]
    color_enc       = color_map[color]
    clarity_enc     = clarity_map[clarity]

    return log_carat, cut_enc, color_enc, clarity_enc, volume, dimension_ratio

# ── UI ────────────────────────────────────────────────────
st.title('💎 Diamond Dynamics')
st.markdown('### Price Prediction & Market Segmentation')
st.markdown('---')

# ── Input Form ────────────────────────────────────────────
st.subheader('🔷 Enter Diamond Attributes')

col1, col2 = st.columns(2)

with col1:
    carat   = st.number_input('Carat',   min_value=0.1,  max_value=5.0,  value=0.5,  step=0.01)
    depth   = st.number_input('Depth %', min_value=40.0, max_value=80.0, value=61.5, step=0.1)
    table   = st.number_input('Table %', min_value=40.0, max_value=100.0,value=55.0, step=0.1)
    x       = st.number_input('Length x (mm)', min_value=0.1, max_value=15.0, value=5.0, step=0.01)

with col2:
    cut     = st.selectbox('Cut',     ['Fair','Good','Very Good','Premium','Ideal'])
    color   = st.selectbox('Color',   ['D','E','F','G','H','I','J'])
    clarity = st.selectbox('Clarity', ['IF','VVS1','VVS2','VS1','VS2','SI1','SI2','I1'])
    y       = st.number_input('Width y (mm)',  min_value=0.1, max_value=15.0, value=5.0, step=0.01)
    z       = st.number_input('Depth z (mm)',  min_value=0.1, max_value=15.0, value=3.0, step=0.01)

st.markdown('---')

# ── Predict Buttons ───────────────────────────────────────
col_btn1, col_btn2 = st.columns(2)

log_carat, cut_enc, color_enc, clarity_enc, volume, dimension_ratio = prepare_input(
    carat, cut, color, clarity, depth, table, x, y, z)

# Regression features
reg_features = np.array([[log_carat, cut_enc, color_enc, clarity_enc,
                           depth, table, x, y, z,
                           volume, carat/1, dimension_ratio]])
reg_scaled = scaler.transform(reg_features)

# Cluster features
cluster_features = np.array([[log_carat, cut_enc, color_enc, clarity_enc,
                               depth, table, x, y, z,
                               volume, dimension_ratio]])
cluster_scaled = scaler_cluster.transform(cluster_features)

with col_btn1:
    if st.button('💰 Predict Price', use_container_width=True):
        log_price_inr = reg_model.predict(reg_scaled)[0]
        price_inr     = np.expm1(log_price_inr)
        price_usd     = price_inr / 84.0

        st.success('💰 Predicted Price')
        st.metric('Price in INR', f'₹ {price_inr:,.0f}')
        st.metric('Price in USD', f'$ {price_usd:,.0f}')

with col_btn2:
    if st.button('🔮 Predict Market Segment', use_container_width=True):
        cluster_id   = cluster_model.predict(cluster_scaled)[0]
        cluster_name = cluster_names[cluster_id]

        emoji_map = {
            'Premium Heavy Diamonds':    '💎',
            'Mid-range Balanced Diamonds':'💍',
            'Affordable Small Diamonds': '💰'
        }
        emoji = emoji_map.get(cluster_name, '🔷')

        st.success('🔮 Market Segment')
        st.metric('Cluster', f'{emoji} {cluster_name}')
        st.info(f'Cluster ID: {cluster_id}')

st.markdown('---')
st.caption('💎 Diamond Dynamics | GUVI Data Science Project')