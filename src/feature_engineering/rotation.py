from dataclasses import dataclass, field

from tqdm import tqdm

from .feature_engineering import FE
from ..logger.logger import logger


@dataclass
class Rotation(FE):

    window_sizes: list[int] = field(default_factory=lambda: [100])

    def feature_engineering(self, data: dict) -> dict:
        for window_size in self.window_sizes:
            logger.debug(f"Calculating rotation smoothed with window size {window_size}")
            for sid in tqdm(data.keys()):
                rotation = (data[sid]['anglez']
                            .diff()
                            .abs()
                            .bfill()
                            .clip(upper=10)
                            .rolling(window=window_size, center=True)
                            .median()
                            .ffill()
                            .bfill()
                            .reset_index(0, drop=True))
                data[sid][f'f_rotation_{window_size}'] = rotation.astype('float32')
        return data
