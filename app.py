import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ─── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Predictor Pro",
    page_icon="🌫️",
    layout="wide"
)

# ─── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px;
            width: 100%;
        }
        h1 { color: #4CAF50; }
    </style>
""", unsafe_allow_html=True)

# ─── Load & Train Model ───────────────────────────────────────────────
@st.cache_resource
def load_data_and_train():
    df = pd.read_csv('city_day.csv')

    cols = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3']

    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    df = df.dropna(subset=['AQI'])
    df = df.dropna(subset=cols)

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    X = df[cols]
    y = df['AQI']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))

    return model, df, score

# ─── AQI Category ─────────────────────────────────────────────────────
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good 🟢"
    elif aqi <= 100:
        return "Satisfactory 🟡"
    elif aqi <= 200:
        return "Moderate 🟠"
    elif aqi <= 300:
        return "Poor 🔴"
    elif aqi <= 400:
        return "Very Poor 🟣"
    else:
        return "Severe ⚫"

# ─── Load Data ────────────────────────────────────────────────────────
model, df, model_score = load_data_and_train()

# ─── Header ───────────────────────────────────────────────────────────
st.title("🌫️ AQI Predictor Pro")
st.markdown("Predict Air Quality Index using ML")
st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Predict AQI", "City Analysis", "Forecast"])

# ════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════
with tab1:
    st.subheader("Enter Pollution Values")

    pm25 = st.number_input("PM2.5", 0.0, 500.0, 60.0)
    pm10 = st.number_input("PM10", 0.0, 500.0, 100.0)
    no = st.number_input("NO", 0.0, 200.0, 10.0)
    no2 = st.number_input("NO2", 0.0, 200.0, 20.0)
    nox = st.number_input("NOx", 0.0, 300.0, 25.0)
    nh3 = st.number_input("NH3", 0.0, 200.0, 15.0)
    co = st.number_input("CO", 0.0, 50.0, 1.0)
    so2 = st.number_input("SO2", 0.0, 200.0, 15.0)
    o3 = st.number_input("O3", 0.0, 300.0, 40.0)

    if st.button("Predict AQI"):
        inp = np.array([[pm25, pm10, no, no2, nox, nh3, co, so2, o3]])
        pred = model.predict(inp)[0]

        st.metric("AQI", f"{pred:.1f}")
        st.success(get_aqi_category(pred))

# ════════════════════════════════════════════
# TAB 2
# ════════════════════════════════════════════
with tab2:
    st.subheader("City AQI Analysis")

    cities = sorted(df['City'].dropna().unique())
    city = st.selectbox("Select City", cities)

    city_df = df[df['City'] == city].dropna().sort_values('Date')

    fig, ax = plt.subplots()
    ax.plot(city_df['Date'], city_df['AQI'])
    ax.set_title(city)
    st.pyplot(fig)

# ════════════════════════════════════════════
# TAB 3 (FIXED)
# ════════════════════════════════════════════
with tab3:
    st.subheader("7-Day Forecast")

    city_f = st.selectbox("Select City", sorted(df['City'].dropna().unique()), key="f")

    features = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3']

    city_data = df[df['City'] == city_f].dropna().sort_values('Date')

    if city_data.empty:
        st.warning("No data available for this city.")
    else:
        last_row = city_data[features].iloc[-1].values

        forecast = []
        for i in range(7):
            noise = np.random.uniform(-0.05, 0.05, len(last_row))
            sim = last_row * (1 + noise)
            pred = model.predict([sim])[0]
            forecast.append(round(pred, 1))

        st.line_chart(forecast)