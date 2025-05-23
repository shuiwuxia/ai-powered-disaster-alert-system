import streamlit as st
from utils.weather_api import fetch_weather
import pandas as pd
import pickle
import os

# Get the absolute path to the data file
base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up from /app/ to project root
csv_path = os.path.join(base_dir, "data", "sample.csv")

# Load the CSV
df = pd.read_csv(csv_path)

print("Data loaded successfully:")
print(df)

# Alert rule
def check_for_flood(row):
    if row["rainfall"] > 30 and row["humidity"] > 85 and row["wind_speed"] > 30:
        return "Flood Risk"
    else:
        return "Safe"

# Apply rule
df["alert"] = df.apply(check_for_flood, axis=1)

print("\n Alert Results:")
print(df[["date", "alert"]])
# Load model and label encoder
model = pickle.load(open("models/model.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

st.title("Live Weather-Based Disaster Alert System")

city = st.text_input("Enter a city name", "Hyderabad")

if st.button("Check Live Weather & Predict Alert"):
    weather = fetch_weather(city)
    
    if weather:
        st.success(" Live weather fetched!")
        st.write(" Weather Data:", weather)

        input_df = pd.DataFrame([weather])
        prediction = model.predict(input_df)[0]
        alert = label_encoder.inverse_transform([prediction])[0]

        st.subheader(f"Predicted Alert: {alert}")
    else:
        st.error("Failed to fetch data. Check city name or internet.")

