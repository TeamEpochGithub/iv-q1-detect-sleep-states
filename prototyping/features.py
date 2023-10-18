import pandas as pd


def add_features(window_data: pd.DataFrame):
    """Return only features, does inplace modifications"""
    abs_diff = window_data['anglez'].diff().abs()
    slope = abs_diff.clip(upper=abs_diff.mean() * 2)
    window_data['median_rotation'] = (slope
                                      .rolling(100, center=True)
                                      .median()
                                      .ffill()
                                      .bfill())

    window_data['enmo_std_1000'] = (window_data['enmo'].rolling(1000, center=True)
                                    .std()
                                    .ffill()
                                    .bfill())
    window_data['enmo_mean_120'] = (window_data['enmo'].rolling(120, center=True)
                                    .mean()
                                    .ffill()
                                    .bfill())
    return window_data[['enmo', 'anglez', 'median_rotation', 'enmo_std_1000', 'enmo_mean_120', 'hour', 'minute']]
