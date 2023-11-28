from datetime import datetime
from pathlib import Path

import hydra
import pandas as pd
from meteostat import Point, Hourly
from omegaconf import DictConfig

from src.logger.logger import logger
from weather_data_downloader.preprocessing import preprocess_weather_data


def download_weather_data(cfg: DictConfig) -> pd.DataFrame:
    """Download the weather data, preprocess it, and save it as a CSV file.

    :param cfg: the Hydra configuration with the start date, end date, location, and filename to save the data to
    """
    # Parse config
    start_date = datetime.strptime(cfg.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(cfg.end_date, '%Y-%m-%d')
    location = Point(cfg.location.latitude, cfg.location.longitude)

    logger.info(f"Downloading weather data for {cfg.location.latitude}, {cfg.location.longitude} "
                f"from {start_date} to {end_date}")

    # Download the weather data
    data: Hourly = Hourly(location, start_date, end_date, timezone=cfg.location.timezone)
    weather_df: pd.DataFrame = data.fetch()

    return weather_df


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    weather_df: pd.DataFrame = download_weather_data(cfg)

    # Preprocess the weather data
    logger.info("Preprocessing weather data")
    weather_df = preprocess_weather_data(weather_df)

    # Save data to a CSV file
    filename = Path().resolve() / cfg.filename
    logger.info(f"Saving weather data to {filename}")

    weather_df.to_csv(filename)


if __name__ == '__main__':
    main()
