import pandas as pd
import os
from feature_engineering.fe import FE


class ExampleFeatureEngineering(FE):
    def run(self, path="data/raw/test_series.parquet"):
        data = None

        target_dir = "data/features/example_feature.parquet"
        # Check if file exists at target_dir
        if os.path.exists(target_dir):
            # If file exists, load it
            print("Loading from parquet")
            data = pd.read_parquet(target_dir)
        else:
            # If file does not exist, create it
            print("Creating parquet")
            data = pd.read_parquet(path)

            # Save file to parquet
            data.to_parquet(target_dir)
