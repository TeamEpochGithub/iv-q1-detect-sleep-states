import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from src.configs.load_config import ConfigException
from src.logger import logger


def standardize(df: pd.DataFrame, method: str) -> None:
    # Select columns that start with 'f_' for standardization
    data_to_standardize = df.filter(regex='^f_')

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        logger.critical("Standardization method not found: " + method)
        raise ConfigException("Standardization method not found: " + method)
    
    # Standardize data
    data_to_standardize = scaler.fit_transform(data_to_standardize)
    df.loc[:, data_to_standardize.columns] = data_to_standardize

    return df
