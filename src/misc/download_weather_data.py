from datetime import datetime
from pathlib import Path

# Meteostat is not added to requirements.txt because it is not available on Kaggle
from meteostat import Point, Hourly

START_DATE = datetime(2017, 1, 1)
END_DATE = datetime(2022, 12, 31)
LOCATION = Point(40.730610, -73.935242)  # New York City
FILE_NAME = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'weather.csv'


def download_weather_data() -> None:
    """Download the weather data and save it to a CSV file"""
    data = Hourly(LOCATION, START_DATE, END_DATE)
    data = data.fetch()

    # Save data to a CSV file
    data.to_csv(FILE_NAME)


if __name__ == "__main__":
    download_weather_data()
