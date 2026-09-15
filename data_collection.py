import requests
import pandas as pd


def fetch_chennai_weather():
    print("Fetching 5 years of historical weather data for Chennai...")

    url = "https://archive-api.open-meteo.com/v1/archive"

    # Coordinates for Chennai, Tamil Nadu
    params = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "start_date": "2019-01-01",
        "end_date": "2023-12-31",
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "rain",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m"
        ],
        "timezone": "Asia/Kolkata"
    }

    print("Sending request to Open-Meteo...")

    response = requests.get(url, params=params)

    # Stop if the API request failed
    response.raise_for_status()

    data = response.json()

    # Build DataFrame
    df = pd.DataFrame({
        "timestamp": data["hourly"]["time"],
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "pressure": data["hourly"]["surface_pressure"],
        "rain": data["hourly"]["rain"],
        "cloud_cover": data["hourly"]["cloud_cover"],
        "wind_speed": data["hourly"]["wind_speed_10m"],
        "wind_direction": data["hourly"]["wind_direction_10m"]
    })

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Save inside data folder
    filename = "data/chennai_weather_2019_2023.csv"

    df.to_csv(filename, index=False)

    print("\nSuccess!")
    print(f"Dataset saved as: {filename}")
    print(f"Total hourly records: {len(df)}")

    print("\nFirst 3 rows:")
    print(df.head(3))


if __name__ == "__main__":
    fetch_chennai_weather()