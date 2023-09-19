import pandas as pd
import os

def example_feature(path="data/raw/test_series.parquet"):
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
    print(data.head())

example_feature()