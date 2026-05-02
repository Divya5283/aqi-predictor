import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ─── Load & Train Model ───────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv('city_day.csv')

    # Clean data
    cols_to_fill = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
    for col in cols_to_fill:
        if col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)

    df.dropna(subset=['AQI'], inplace=True)

    features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
    X = df[features]
    y = df['AQI']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

# ─── AQI Category Helper ──────────────────────────────────────────────
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good 🟢", "green"
    elif aqi <= 100:
        return "Satisfactory 🟡", "yellow"
    elif aqi <= 200:
        return "Moderate 🟠", "orange"
    elif aqi <= 300:
        return "Poor 🔴", "red"
    elif aqi <= 400:
        return "Very Poor 🟣", "purple"
    else:
        return "Severe ⚫", "black"

# ─── UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="centered")

st.title("🌫️ Air Quality Index (AQI) Predictor")
st.markdown("Enter pollutant levels below to predict the AQI for a location.")
st.divider()

model = train_model()

# Input sliders
col1, col2, col3 = st.columns(3)

with col1:
    pm25  = st.number_input("PM2.5 (µg/m³)",  min_value=0.0, max_value=500.0, value=60.0)
    no    = st.number_input("NO (µg/m³)",      min_value=0.0, max_value=200.0, value=10.0)
    nh3   = st.number_input("NH3 (µg/m³)",     min_value=0.0, max_value=200.0, value=15.0)

with col2:
    pm10  = st.number_input("PM10 (µg/m³)",    min_value=0.0, max_value=500.0, value=100.0)
    no2   = st.number_input("NO2 (µg/m³)",     min_value=0.0, max_value=200.0, value=20.0)
    co    = st.number_input("CO (mg/m³)",       min_value=0.0, max_value=50.0,  value=1.0)

with col3:
    nox   = st.number_input("NOx (µg/m³)",     min_value=0.0, max_value=300.0, value=25.0)
    so2   = st.number_input("SO2 (µg/m³)",     min_value=0.0, max_value=200.0, value=15.0)
    o3    = st.number_input("O3 (µg/m³)",      min_value=0.0, max_value=300.0, value=40.0)

st.divider()

# Predict button
if st.button("🔍 Predict AQI", use_container_width=True):
    input_data = np.array([[pm25, pm10, no, no2, nox, nh3, co, so2, o3]])
    prediction = model.predict(input_data)[0]
    category, color = get_aqi_category(prediction)

    st.markdown(f"### Predicted AQI: `{prediction:.1f}`")
    st.markdown(f"### Category: **:{color}[{category}]**")

    # Health advice
    st.divider()
    st.markdown("#### 💡 Health Advice")
    if prediction <= 50:
        st.success("Air quality is great! Perfect for outdoor activities.")
    elif prediction <= 100:
        st.info("Air quality is acceptable. Sensitive individuals should take care.")
    elif prediction <= 200:
        st.warning("Moderate pollution. Reduce prolonged outdoor activity.")
    elif prediction <= 300:
        st.error("Poor air quality. Avoid outdoor activities if possible.")
    else:
        st.error("🚨 Severe pollution! Stay indoors and use air purifiers.")