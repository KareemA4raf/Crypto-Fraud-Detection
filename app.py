import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="ETH Fraud Detector", page_icon="🛡️", layout="wide")

st.title("🛡️ Ethereum Fraud Detection Dashboard")
st.markdown("### Capstone Project: AI-Powered Blockchain Security")
st.markdown("---")

@st.cache_resource
def load_all_assets():
    scaler = joblib.load('scaler.pkl')
    models = {
        "XGBoost": joblib.load('xgboost_model.pkl'),
        "LightGBM": joblib.load('lightgbm_model.pkl'),
        "Random Forest": joblib.load('random_forest_model.pkl'),
        "Logistic Regression": joblib.load('logistic_regression_model.pkl'),
        "KNN": joblib.load('knn_model.pkl')
    }
    dnn = load_model('tensorflow_dnn.keras')
    return scaler, models, dnn, scaler.feature_names_in_

try:
    scaler, ml_models, dnn_model, expected_features = load_all_assets()
    assets_ready = True
except Exception as e:
    st.error(f"Error loading assets: {e}")
    assets_ready = False

def preprocess_input(df, feat_list):
    df.columns = df.columns.str.replace(' ', '_')
    df_clean = df.drop(columns=['FLAG','Unnamed:_0','Index','Address'], errors='ignore')
    df_clean = pd.get_dummies(df_clean).fillna(0)
    missing = list(set(feat_list) - set(df_clean.columns))
    if missing:
        df_missing = pd.DataFrame(0, index=df_clean.index, columns=missing)
        df_clean = pd.concat([df_clean, df_missing], axis=1)
    return df_clean[feat_list].copy()

tab1, tab2, tab3 = st.tabs(["📊 Dataset", "🏆 Performance", "🔍 Live Detection"])

df_raw = pd.read_csv('transaction_dataset.csv') if os.path.exists('transaction_dataset.csv') else None

with tab1:
    if df_raw is not None:
        st.subheader("Ethereum Transaction Samples")
        st.dataframe(df_raw.head(50), width='stretch')
        st.plotly_chart(px.pie(df_raw, names='FLAG', hole=0.4, title="Class Distribution"))

with tab2:
    st.header("Model Performance Metrics")
    metrics = pd.DataFrame({
        "Model": ["XGBoost", "LightGBM", "Random Forest", "Logistic Regression", "KNN", "Safety Ensemble"],
        "Accuracy": ["99.49%", "99.44%", "99.09%", "97.77%", "96.65%", "99.58%"],
        "Recall (Fraud)": ["97.71%", "97.71%", "95.87%", "96.56%", "95.18%", "98.50%"],
        "Complexity": ["High", "High", "Medium", "Simple", "Medium", "Ensemble"]
    })
    st.table(metrics)
    st.success("The Safety Ensemble is tuned for high sensitivity to capture potential threats.")

with tab3:
    if df_raw is not None and assets_ready:
        if st.button("🎲 Fetch Random Transaction", type="primary", width='stretch'):
            random_idx = np.random.randint(0, len(df_raw))
            raw_row = df_raw.iloc[[random_idx]]
            actual = df_raw.iloc[random_idx]['FLAG']
            
            st.subheader(f"Analyzing Transaction #{random_idx}")
            st.dataframe(raw_row, width='stretch')
            
            processed = preprocess_input(raw_row, expected_features)
            scaled = scaler.transform(processed)
            scaled_df = pd.DataFrame(scaled, columns=expected_features)
            
            st.markdown(f"**Ground Truth:** {'🚨 FRAUD' if actual==1 else '✅ SAFE'}")
            st.markdown("---")
            
            st.markdown("### Multi-Architecture Prediction Showdown")
            
            all_predictions = []
            cols = st.columns(7)
            
            for i, (name, m) in enumerate(ml_models.items()):
                pred = m.predict(scaled_df)[0]
                all_predictions.append(pred)
                with cols[i]:
                    st.markdown(f"**{name}**")
                    if pred == 1: st.error("🚨 FRAUD")
                    else: st.success("✅ SAFE")
            
            prob = dnn_model.predict(scaled, verbose=0)[0][0]
            tf_pred = 1 if prob > 0.4 else 0
            all_predictions.append(tf_pred)
            with cols[5]:
                st.markdown("**TF DNN**")
                if tf_pred == 1: st.error("🚨 FRAUD")
                else: st.success("✅ SAFE")

            fraud_votes = sum(all_predictions)
            
            with cols[6]:
                st.markdown("**Safety Ensemble**")
                if fraud_votes >= 2: 
                    st.error("🚨 FRAUD")
                    st.caption(f"Flagged by {fraud_votes} models")
                else:
                    st.success("✅ SAFE")
                    st.caption("Consensus reached")
                    