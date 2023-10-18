import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from src.configs.load_config import ConfigException
from src.logger import logger


def standardize(df: pd.DataFrame, method: str) -> pd.DataFrame:
    # Select columns that start with 'f_' for standardization
    # data_to_standardize = df.filter(regex='^f_')
    # if data_to_standardize.empty:
    #     return df
    data_to_standardize = df[['enmo', 'anglez', 'f_hour', 'f_minute']]

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    elif method == "none":
        return df
    else:
        logger.critical("Standardization method not found: " + method)
        raise ConfigException("Standardization method not found: " + method)

    # Standardize data
    standardized_data = scaler.fit_transform(data_to_standardize)
    df.loc[:, data_to_standardize.columns] = standardized_data

    return df
