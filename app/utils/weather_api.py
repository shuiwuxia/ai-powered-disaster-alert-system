import requests

API_KEY = 'd02a952366f9b124752aba4bbc24312a'

def fetch_weather(city="Hyderabad"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(url, timeout=5)  # Set timeout to 5 seconds
        response.raise_for_status()
        
        data = response.json()
        weather_data = {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "rainfall": data.get("rain", {}).get("1h", 0.0),
            "wind_speed": data["wind"]["speed"] * 3.6  # convert m/s to km/h
        }
        return weather_data

    except requests.exceptions.ReadTimeout:
        print(f"Request to {city} timed out.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Test the function
print(fetch_weather("Hyderabad"))
