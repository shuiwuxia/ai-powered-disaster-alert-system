import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import os
from datetime import datetime
from utils.weather_api import fetch_weather
import sys
import yagmail
import joblib
import re

st.title("AI-Powered Disaster Alert System")
st.markdown("Real-time weather and disaster alerts from ARR keep you updated!")
st_autorefresh(interval=300000, key="refresh")

st.sidebar.header("Weather Search")
search_city = st.sidebar.text_input("Enter city name").strip().lower()
cities = ["Hyderabad", "Raichur", "Secunderabad", "Mirzapur", "Bangalore", "Delhi", "Mumbai", "Chennai", "Kolkata"]

city_dict = {city.lower(): city for city in cities}
ALERT_CONDITIONS = {
    "Flood": lambda row: row["rainfall"] > 30 and row["humidity"] > 85 and row["wind_speed"] > 30,
    "Heatwave": lambda row: row["temperature"] >= 40 and row["humidity"] < 30,
    "Cyclone": lambda row: row["wind_speed"] > 50 and row["rainfall"] > 20
}
def detect_alerts(row):
    alerts = []
    if row.get("rainfall", 0) > 30:
        if row.get("humidity", 0) > 85 and row.get("wind_speed", 0) > 30:
            alerts.append("Flood")
    if row.get("temperature", 0) >= 40:
        if row.get("humidity", 0) < 30:
            alerts.append("Heatwave")
    if row.get("wind_speed", 0) > 50:
        if row.get("rainfall", 0) > 20:
            alerts.append("Cyclone")
    return ", ".join(alerts) if alerts else "Safe"
def alertsmail(city, alert_type):
    yag = yagmail.SMTP("ranvithareddy.a@gmail.com", "mrxl pnql cjqq zvfa") 
    subject = f" {alert_type} Alert in {city}"
    body = f"A {alert_type} has been detected in {city}."
    yag.send("ranvithareddy.a@gmail.com", subject, body)

if search_city in city_dict:
    city = city_dict[search_city]
    weather = fetch_weather(city)
    st.subheader(f"Weather Report for {city}")
    for key, label in zip(["temperature", "humidity", "rainfall", "wind_speed"],
                           ["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "Wind Speed (km/h)"]):
        st.metric(label, f"{weather[key]}")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.write(f"Alert: {detect_alerts(weather)}")

data = []
for city in cities:
    w = fetch_weather(city)
    w.update({"date": datetime.now().strftime('%Y-%m-%d'), "city": city})
    data.append(w)

df = pd.DataFrame(data)
for col in ["temperature", "humidity", "rainfall", "wind_speed"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df["Alert"] = df.apply(detect_alerts, axis=1)

alerts_sent = set()
for _, row in df.iterrows():
    if row["Alert"] != "Safe":
        key = (row["city"], row["Alert"])
        if key not in alerts_sent:
            alertsmail(row["city"], row["Alert"])
            alerts_sent.add(key)

@st.cache_resource
def load_model():
    return joblib.load('models/disaster_model.pkl'), joblib.load('models/tfidf_vectorizer.pkl')

model, vectorizer = load_model()

st.subheader("Live Weather Report")
st.dataframe(df)

if not os.path.exists("data/weather_history.csv"):
    df.to_csv("data/weather_history.csv", index=False)
else:
    existing = pd.read_csv("data/weather_history.csv")
    updated = pd.concat([existing, df]).drop_duplicates(subset=["date", "city"])
    updated.to_csv("data/weather_history.csv", index=False)

st.download_button(
    "Download Weather Data as CSV",
    data=df.to_csv(index=False),
    file_name="today_weather.csv",
    mime="text/csv"
)
st.subheader("Risk Alerts")
for alert_type in ALERT_CONDITIONS:
    filtered = df[df["Alert"].str.contains(alert_type)]
    if not filtered.empty:
        st.error(f"{alert_type} detected in {len(filtered)} cities:")
        st.dataframe(filtered[["date", "city", "Alert"]])
    else:
        st.success(f"No {alert_type} risks detected.")
st.subheader("Tweet / News Classification")
input_text = st.text_area("Paste a tweet or news headline:", height=100)

if st.button("Classify"):
    if input_text.strip():
        def clean(text):
            text = text.lower()
            text = re.sub(r"http\S+|www\S+|https\S+", "", text)
            text = re.sub(r"@\w+|\#", "", text)
            text = re.sub(r"[^\w\s]", "", text)
            text = re.sub(r"\d+", "", text)
            return text.strip()

        cleaned = clean(input_text)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        label_map = {0: "Flood", 1: "Cyclone", 2: "Heatwave", 3: "None"}
        st.info(f"Prediction: **{label_map[prediction]}**")
    else:
        st.warning("Please enter some text to classify.")
st.sidebar.title("App Controls")
if st.sidebar.button("Shutdown App"):
    st.success("Shutting down... Goodbye!")
    st.stop()
    sys.exit()

st.markdown("Developed by Ranvitha Reddy.A(AD24B1004)")
