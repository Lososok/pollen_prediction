import pandas as pd
import requests

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 55.7558,
    "longitude": 37.6176,
    "start_date": "2017-01-01",
    "end_date": "2024-01-01",
    "hourly": "temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m",
    "timezone": "Europe/Moscow"
}
response = requests.get(url, params=params)
data = response.json()

response = requests.get(url, params=params)
weather_data = response.json()

df = pd.DataFrame({
    "time": weather_data["hourly"]["time"],
    "temperature": weather_data["hourly"]["temperature_2m"],
    "precipitation": weather_data["hourly"]["precipitation"],
    "humidity": weather_data["hourly"]["relative_humidity_2m"],
    "wind_speed": weather_data["hourly"]["wind_speed_10m"]
})

df.to_csv("moscow_weather_2017-2024.csv", index=False)