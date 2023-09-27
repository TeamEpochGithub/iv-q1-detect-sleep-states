"""Integration test for the ZipTrainEvents preprocessing step.

It is not a unit test since it requires the train dataset and labels.
"""
import pandas as pd

from src.configs.load_config import ConfigLoader
from src.preprocessing.zip_train_events import ZipTrainEvents

if __name__ == '__main__':
    zip_train_events = ZipTrainEvents()

    # Load training dataset
    config = ConfigLoader("test/test_config.json")
    train_series = pd.read_parquet(config.get_pp_in() + "/train_series.parquet").head(30000)

    zipped = zip_train_events.preprocess(train_series)
    print(zipped[zipped["event"].isin(["onset", "wakeup"])])

    onsets = zipped[zipped["event"] == "onset"]
    wakeups = zipped[zipped["event"] == "wakeup"]

    assert (onsets.shape[0] > 0)
    assert (onsets.shape[0] == wakeups.shape[0])
