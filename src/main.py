import pandas as pd
import numpy as np

def run(test_series_path):
    test = pd.read_parquet(test_series_path)
    submission = test[["series_id",'step']].copy()
    submission['event'] = 'onset'
    submission['score'] = np.random.uniform(0,1,len(submission))
    submission = submission.reset_index(drop=True).reset_index(names="row_id")
    submission.to_csv("submission.csv",index=False)