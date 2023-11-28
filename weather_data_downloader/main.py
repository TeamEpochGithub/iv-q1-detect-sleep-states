from datetime import datetime
from pathlib import Path

import hydra
from meteostat import Point, Hourly
from omegaconf import DictConfig

from src.logger.logger import logger


@hydra.main(version_base=None, config_path='conf', config_name='config')
def download_weather_data(cfg: DictConfig) -> None:
    """Download the weather data and save it to a CSV file

    :param cfg: the Hydra configuration with the start date, end date, location, and filename to save the data to
    """
    # Parse config
    start_date = datetime.strptime(cfg.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(cfg.end_date, '%Y-%m-%d')
    location = Point(cfg.location.latitude, cfg.location.longitude)
    filename = Path().resolve() / cfg.filename

    logger.info(f"Downloading weather data for {cfg.location.latitude}, {cfg.location.longitude} "
                f"from {start_date} to {end_date}")

    # Download the weather data
    data = Hourly(location, start_date, end_date, timezone=cfg.location.timezone)
    data = data.fetch()

    logger.info(f"Saving weather data to {filename}")

    # Save data to a CSV file
    data.to_csv(filename)


if __name__ == '__main__':
    download_weather_data()
