import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import LabelEncoder
import pycountry
import os

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Anomalyze — Fraud Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.main { background-color: #0F1117; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1A1D27, #22263a);
    border: 1px solid #2e3250;
    border-radius: 12px;
    padding: 16px 20px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #1A1D27;
    border-right: 1px solid #2e3250;
}

/* Buttons */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #E24B4A, #c0392b);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 28px;
    font-size: 16px;
    font-weight: 600;
    width: 100%;
    transition: all 0.3s ease;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #c0392b, #E24B4A);
    transform: translateY(-1px);
}

/* Input fields */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div {
    background-color: #1A1D27;
    border: 1px solid #2e3250;
    border-radius: 8px;
    color: #FAFAFA;
}

/* Section headers */
.section-header {
    background: linear-gradient(135deg, #1A1D27, #22263a);
    border-left: 4px solid #E24B4A;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 16px 0;
    font-size: 16px;
    font-weight: 600;
    color: #FAFAFA;
}

/* Logo area */
.logo-container {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 0 20px 0;
}
.logo-text {
    font-size: 24px;
    font-weight: 800;
    background: linear-gradient(135deg, #E24B4A, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.logo-sub {
    font-size: 11px;
    color: #888;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Prediction result cards */
.result-fraud {
    background: linear-gradient(135deg, #2d1a1a, #3d1f1f);
    border: 2px solid #E24B4A;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.result-safe {
    background: linear-gradient(135deg, #1a2d1a, #1f3d1f);
    border: 2px solid #1D9E75;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.result-title {
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 8px;
}
.result-subtitle {
    font-size: 15px;
    opacity: 0.8;
}

/* Form card */
.form-card {
    background: #1A1D27;
    border: 1px solid #2e3250;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}

/* Divider */
.custom-divider {
    height: 1px;
    background: linear-gradient(to right, #E24B4A, transparent);
    margin: 20px 0;
}

/* Nav pills */
.nav-pill {
    background: #22263a;
    border-radius: 8px;
    padding: 8px 16px;
    margin: 4px 0;
    cursor: pointer;
    transition: all 0.2s;
}
</style>
""", unsafe_allow_html=True)

# ─── Paths ─────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, 'models', 'fraud_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'models', 'feature_names.pkl')
DATA_PATH     = os.path.join(BASE_DIR, 'data', 'Fraud Detection Dataset.csv')

# ─── World Countries List ──────────────────────────────────
ALL_COUNTRIES = sorted([country.name for country in pycountry.countries])

# ─── Load Model & Data ─────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    model, feature_names = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Data loading error: {e}")
    st.stop()

# ─── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-container">
        <div>
            <div class="logo-text">🧠 Anomalyze</div>
            <div class="logo-sub">Fraud Detection System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    page = st.radio("", ["📊  Dashboard", "🔍  Predict Transaction", "📁  Data Explorer"],
                    label_visibility="collapsed")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    total = len(df_raw)
    fraud = int(df_raw['Fraudulent'].sum())
    st.markdown(f"""
    <div style='font-size:12px; color:#888; line-height:2;'>
        <div>Total records: <b style='color:#FAFAFA'>{total:,}</b></div>
        <div>Fraud cases: <b style='color:#E24B4A'>{fraud:,}</b></div>
        <div>Model: <b style='color:#1D9E75'>XGBoost</b></div>
        <div>Status: <b style='color:#1D9E75'>● Live</b></div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "📊  Dashboard":

    st.markdown("<h1 style='font-size:32px; font-weight:800;'>📊 Dashboard Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;'>Real-time fraud analytics across all transactions</p>", unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    total     = len(df_raw)
    fraud     = int(df_raw['Fraudulent'].sum())
    normal    = total - fraud
    fraud_pct = (fraud / total) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🗂️ Total Transactions", f"{total:,}")
    col2.metric("🚨 Fraud Cases",        f"{fraud:,}",  delta=f"{fraud_pct:.1f}% rate", delta_color="inverse")
    col3.metric("✅ Normal Cases",       f"{normal:,}")
    col4.metric("📈 Fraud Rate",         f"{fraud_pct:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Chart styling
    plt.rcParams.update({
        'figure.facecolor': '#1A1D27',
        'axes.facecolor':   '#1A1D27',
        'axes.labelcolor':  '#FAFAFA',
        'xtick.color':      '#888',
        'ytick.color':      '#888',
        'text.color':       '#FAFAFA',
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.spines.left':   False,
        'axes.spines.bottom': False,
    })

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Transaction Split</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        wedges, texts, autotexts = ax.pie(
            [normal, fraud],
            labels=['Normal', 'Fraud'],
            colors=['#1D9E75', '#E24B4A'],
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(width=0.6),
            textprops={'fontsize': 11, 'color': '#FAFAFA'}
        )
        for at in autotexts:
            at.set_color('white')
            at.set_fontweight('bold')
        ax.set_title('Fraud vs Normal', fontsize=13, fontweight='bold', color='#FAFAFA')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">Fraud by Payment Method</div>', unsafe_allow_html=True)
        fraud_by_payment = df_raw[df_raw['Fraudulent'] == 1]['Payment_Method'].value_counts().dropna().head(6)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.barh(fraud_by_payment.index, fraud_by_payment.values,
                       color='#E24B4A', alpha=0.85, height=0.6)
        for bar, val in zip(bars, fraud_by_payment.values):
            ax.text(val + 5, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', fontsize=10, color='#FAFAFA')
        ax.set_title('Payment Method', fontsize=13, fontweight='bold', color='#FAFAFA')
        ax.set_facecolor('#1A1D27')
        fig.patch.set_facecolor('#1A1D27')
        st.pyplot(fig)
        plt.close()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Fraud by Device</div>', unsafe_allow_html=True)
        fraud_by_device = df_raw[df_raw['Fraudulent'] == 1]['Device_Used'].value_counts().dropna()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(fraud_by_device.index, fraud_by_device.values,
               color='#378ADD', alpha=0.85, edgecolor='none', width=0.5)
        ax.set_title('Device Used', fontsize=13, fontweight='bold', color='#FAFAFA')
        plt.xticks(rotation=30, ha='right')
        fig.patch.set_facecolor('#1A1D27')
        ax.set_facecolor('#1A1D27')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">Amount Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        df_raw[df_raw['Fraudulent'] == 0]['Transaction_Amount'].dropna().hist(
            bins=40, alpha=0.6, color='#1D9E75', label='Normal', ax=ax)
        df_raw[df_raw['Fraudulent'] == 1]['Transaction_Amount'].dropna().hist(
            bins=40, alpha=0.7, color='#E24B4A', label='Fraud', ax=ax)
        ax.set_title('Transaction Amounts', fontsize=13, fontweight='bold', color='#FAFAFA')
        ax.legend(facecolor='#1A1D27', labelcolor='#FAFAFA')
        fig.patch.set_facecolor('#1A1D27')
        ax.set_facecolor('#1A1D27')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT TRANSACTION
# ══════════════════════════════════════════════════════════════
elif page == "🔍  Predict Transaction":

    st.markdown("<h1 style='font-size:32px; font-weight:800;'>🔍 Predict Transaction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;'>Enter transaction details to check if it is fraudulent</p>", unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-header">💰 Transaction Info</div>', unsafe_allow_html=True)
        transaction_amount   = st.number_input("Amount (₹)", min_value=0.0, value=1000.0, step=100.0)
        transaction_type     = st.selectbox("Transaction Type", df_raw['Transaction_Type'].dropna().unique().tolist())
        payment_method       = st.selectbox("Payment Method",   df_raw['Payment_Method'].dropna().unique().tolist())
        time_of_transaction  = st.slider("Time of Transaction (hour)", min_value=0, max_value=23, value=12)

    with col2:
        st.markdown('<div class="section-header">📱 Device & Location</div>', unsafe_allow_html=True)
        device_used = st.selectbox("Device Used", df_raw['Device_Used'].dropna().unique().tolist())
        location    = st.selectbox("Country / Location", ALL_COUNTRIES, index=ALL_COUNTRIES.index("India") if "India" in ALL_COUNTRIES else 0)

    with col3:
        st.markdown('<div class="section-header">👤 Account Details</div>', unsafe_allow_html=True)
        previous_fraud       = st.number_input("Previous Fraud History", min_value=0, value=0, step=1,
                                                help="How many times this account was involved in fraud before")
        account_age          = st.number_input("Account Age (days)",     min_value=0, value=365, step=1,
                                                help="How old is this account in days")
        num_transactions_24h = st.number_input("Transactions in Last 24H", min_value=0, value=1, step=1,
                                                help="Number of transactions this account made in last 24 hours")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Analyze Transaction for Fraud", use_container_width=True)

    if predict_btn:
        def encode_value(column, value):
            le = LabelEncoder()
            le.fit(df_raw[column].dropna())
            try:
                return int(le.transform([value])[0])
            except:
                return 0

        # For location — encode based on dataset locations
        location_cols = df_raw['Location'].dropna().unique().tolist()
        if location in location_cols:
            encoded_location = encode_value('Location', location)
        else:
            encoded_location = 0

        input_data = pd.DataFrame([{
            'Transaction_Amount':               transaction_amount,
            'Transaction_Type':                 encode_value('Transaction_Type', transaction_type),
            'Time_of_Transaction':              time_of_transaction,
            'Device_Used':                      encode_value('Device_Used', device_used),
            'Location':                         encoded_location,
            'Previous_Fraudulent_Transactions': previous_fraud,
            'Account_Age':                      account_age,
            'Number_of_Transactions_Last_24H':  num_transactions_24h,
            'Payment_Method':                   encode_value('Payment_Method', payment_method),
        }])

        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[feature_names]

        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center;'>Analysis Result</h3>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
        with res_col2:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-fraud">
                    <div class="result-title" style="color:#E24B4A;">🚨 FRAUD DETECTED</div>
                    <div class="result-subtitle">This transaction looks suspicious</div>
                    <br>
                    <div style="font-size:42px; font-weight:900; color:#E24B4A;">{probability[1]*100:.1f}%</div>
                    <div style="color:#888; font-size:13px;">Fraud Probability</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    <div class="result-title" style="color:#1D9E75;">✅ LEGITIMATE</div>
                    <div class="result-subtitle">This transaction looks safe</div>
                    <br>
                    <div style="font-size:42px; font-weight:900; color:#1D9E75;">{probability[0]*100:.1f}%</div>
                    <div style="color:#888; font-size:13px;">Safe Probability</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("✅ Safe Probability",  f"{probability[0]*100:.1f}%")
        c2.metric("🚨 Fraud Probability", f"{probability[1]*100:.1f}%", delta_color="inverse")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "📁  Data Explorer":

    st.markdown("<h1 style='font-size:32px; font-weight:800;'>📁 Data Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;'>Explore the dataset used to train ShieldAI</p>", unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows",    f"{df_raw.shape[0]:,}")
    col2.metric("Total Columns", f"{df_raw.shape[1]}")
    col3.metric("Fraud Cases",   f"{int(df_raw['Fraudulent'].sum()):,}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Raw Dataset — First 100 Rows</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(100), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            st.dataframe(missing.rename("Missing Count"))
        else:
            st.success("No missing values in dataset!")

    with col2:
        st.markdown('<div class="section-header">Column Data Types</div>', unsafe_allow_html=True)
        st.dataframe(df_raw.dtypes.rename("Data Type"))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.describe(), use_container_width=True)
