import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="💎 Diamond Dynamics",
    page_icon="💎",
    layout="wide"
)

# ── Load Models ───────────────────────────────────────────
@st.cache_resource
def load_models():
    reg_model      = joblib.load('../models/best_regression_model.pkl')
    cluster_model  = joblib.load('../models/kmeans_model.pkl')
    scaler         = joblib.load('../models/scaler.pkl')
    scaler_cluster = joblib.load('../models/scaler_cluster.pkl')
    cluster_names  = joblib.load('../models/cluster_names.pkl')
    return reg_model, cluster_model, scaler, scaler_cluster, cluster_names

reg_model, cluster_model, scaler, scaler_cluster, cluster_names = load_models()

# ── Load Data for Visuals ─────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('../data/diamonds_final.csv')

df = load_data()

# ── Helper ────────────────────────────────────────────────
def prepare_input(carat, cut, color, clarity, depth, table, x, y, z):
    cut_map     = {'Fair':0,'Good':1,'Very Good':2,'Premium':3,'Ideal':4}
    color_map   = {'J':0,'I':1,'H':2,'G':3,'F':4,'E':5,'D':6}
    clarity_map = {'I1':0,'SI2':1,'SI1':2,'VS2':3,'VS1':4,'VVS2':5,'VVS1':6,'IF':7}

    volume          = x * y * z
    dimension_ratio = (x + y) / (2 * z)
    log_carat       = np.log1p(carat)
    price_per_carat = 1.0  # placeholder for prediction
    cut_enc         = cut_map[cut]
    color_enc       = color_map[color]
    clarity_enc     = clarity_map[clarity]

    return log_carat, cut_enc, color_enc, clarity_enc, volume, dimension_ratio

# ── Title ─────────────────────────────────────────────────
st.title('💎 Diamond Dynamics')
st.markdown('### Price Prediction & Market Segmentation')
st.markdown('---')

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(['💰 Price Prediction', '🔮 Market Segment', '📊 Visual Insights'])

# ═══════════════════════════════════════════════════════════
# TAB 1 — Price Prediction
# ═══════════════════════════════════════════════════════════
with tab1:
    st.subheader('💰 Predict Diamond Price')
    st.markdown('Enter diamond attributes to predict its price in INR and USD.')

    col1, col2 = st.columns(2)

    with col1:
        carat = st.number_input('Carat',        min_value=0.1,  max_value=5.0,   value=0.5,  step=0.01)
        depth = st.number_input('Depth %',      min_value=40.0, max_value=80.0,  value=61.5, step=0.1)
        table = st.number_input('Table %',      min_value=40.0, max_value=100.0, value=55.0, step=0.1)
        x     = st.number_input('Length x (mm)',min_value=0.1,  max_value=15.0,  value=5.0,  step=0.01)

    with col2:
        cut     = st.selectbox('Cut',     ['Fair','Good','Very Good','Premium','Ideal'])
        color   = st.selectbox('Color',   ['D','E','F','G','H','I','J'])
        clarity = st.selectbox('Clarity', ['IF','VVS1','VVS2','VS1','VS2','SI1','SI2','I1'])
        y       = st.number_input('Width y (mm)', min_value=0.1, max_value=15.0, value=5.0, step=0.01)
        z       = st.number_input('Depth z (mm)', min_value=0.1, max_value=15.0, value=3.0, step=0.01)

    if st.button('💰 Predict Price', use_container_width=True):
        log_carat, cut_enc, color_enc, clarity_enc, volume, dimension_ratio = prepare_input(
            carat, cut, color, clarity, depth, table, x, y, z)

        price_per_carat_val = 1.0
        reg_features = np.array([[log_carat, cut_enc, color_enc, clarity_enc,
                                   depth, table, x, y, z,
                                   volume, price_per_carat_val, dimension_ratio]])
        reg_scaled = scaler.transform(reg_features)

        log_price_inr = reg_model.predict(reg_scaled)[0]
        price_inr     = np.expm1(log_price_inr)
        price_usd     = price_inr / 84.0

        st.success('✅ Prediction Complete!')
        c1, c2 = st.columns(2)
        c1.metric('💵 Price in USD', f'$ {price_usd:,.0f}')
        c2.metric('💰 Price in INR', f'₹ {price_inr:,.0f}')

        st.info(f'📌 Diamond: {carat}ct | {cut} cut | {color} color | {clarity} clarity')

# ═══════════════════════════════════════════════════════════
# TAB 2 — Market Segment
# ═══════════════════════════════════════════════════════════
with tab2:
    st.subheader('🔮 Predict Market Segment')
    st.markdown('Find out which market category your diamond belongs to.')

    col1, col2 = st.columns(2)

    with col1:
        carat2 = st.number_input('Carat ',        min_value=0.1,  max_value=5.0,   value=0.5,  step=0.01, key='c2')
        depth2 = st.number_input('Depth % ',      min_value=40.0, max_value=80.0,  value=61.5, step=0.1,  key='d2')
        table2 = st.number_input('Table % ',      min_value=40.0, max_value=100.0, value=55.0, step=0.1,  key='t2')
        x2     = st.number_input('Length x (mm) ',min_value=0.1,  max_value=15.0,  value=5.0,  step=0.01, key='x2')

    with col2:
        cut2     = st.selectbox('Cut ',     ['Fair','Good','Very Good','Premium','Ideal'], key='cut2')
        color2   = st.selectbox('Color ',   ['D','E','F','G','H','I','J'],                key='col2')
        clarity2 = st.selectbox('Clarity ', ['IF','VVS1','VVS2','VS1','VS2','SI1','SI2','I1'], key='cla2')
        y2       = st.number_input('Width y (mm) ', min_value=0.1, max_value=15.0, value=5.0, step=0.01, key='y2')
        z2       = st.number_input('Depth z (mm) ', min_value=0.1, max_value=15.0, value=3.0, step=0.01, key='z2')

    if st.button('🔮 Predict Market Segment', use_container_width=True):
        log_carat, cut_enc, color_enc, clarity_enc, volume, dimension_ratio = prepare_input(
            carat2, cut2, color2, clarity2, depth2, table2, x2, y2, z2)

        cluster_feats = np.array([[log_carat, cut_enc, color_enc, clarity_enc,
                                   depth2, table2, x2, y2, z2,
                                   volume, dimension_ratio]])
        cluster_scaled = scaler_cluster.transform(cluster_feats)

        cluster_id   = cluster_model.predict(cluster_scaled)[0]
        cluster_name = cluster_names[cluster_id]

        emoji_map = {
            'Premium Heavy Diamonds':     '💎',
            'Mid-range Balanced Diamonds':'💍',
            'Affordable Small Diamonds':  '💰'
        }
        emoji = emoji_map.get(cluster_name, '🔷')

        color_map2 = {
            'Premium Heavy Diamonds':     'success',
            'Mid-range Balanced Diamonds':'info',
            'Affordable Small Diamonds':  'warning'
        }

        st.success('✅ Segment Identified!')
        st.markdown(f'## {emoji} {cluster_name}')
        st.metric('Cluster ID', f'Cluster {cluster_id}')

        # Cluster description
        desc_map = {
            'Premium Heavy Diamonds':     '💎 Large, expensive, premium-grade stones. Avg price > ₹9,00,000',
            'Mid-range Balanced Diamonds':'💍 Balanced in size and cost. Avg price ~ ₹3,75,000',
            'Affordable Small Diamonds':  '💰 Small, budget-friendly stones. Avg price ~ ₹90,000'
        }
        st.info(desc_map[cluster_name])

# ═══════════════════════════════════════════════════════════
# TAB 3 — Visual Insights
# ═══════════════════════════════════════════════════════════
with tab3:
    st.subheader('📊 Visual Insights')

    # Cluster distribution
    st.markdown('#### 💎 Market Segment Distribution')
    cluster_counts = df['cluster_name'].value_counts()

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    colors = ['gold', 'steelblue', 'coral']
    ax1.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='black')
    ax1.set_title('Diamond Market Segment Distribution')
    ax1.set_xlabel('Market Segment')
    ax1.set_ylabel('Count')
    plt.tight_layout()
    st.pyplot(fig1)

    # Avg price per cluster
    st.markdown('#### 💰 Average Price per Market Segment')
    avg_price = df.groupby('cluster_name')['price_inr'].mean()

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(avg_price.index, avg_price.values, color=colors, edgecolor='black')
    ax2.set_title('Average Price (INR) per Market Segment')
    ax2.set_xlabel('Market Segment')
    ax2.set_ylabel('Avg Price (INR)')
    plt.tight_layout()
    st.pyplot(fig2)

    # Avg carat per cluster
    st.markdown('#### ⚖️ Average Carat per Market Segment')
    avg_carat = df.groupby('cluster_name')['carat'].mean()

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar(avg_carat.index, avg_carat.values, color=colors, edgecolor='black')
    ax3.set_title('Average Carat per Market Segment')
    ax3.set_xlabel('Market Segment')
    ax3.set_ylabel('Avg Carat')
    plt.tight_layout()
    st.pyplot(fig3)

st.markdown('---')
st.caption('💎 Diamond Dynamics | Data Science Project | Random Forest + KMeans')