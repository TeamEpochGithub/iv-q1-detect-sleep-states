from typing import Final

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

WEATHER_CONDITION_CODES: Final[list[str]] = [
    'null', 'Clear', 'Fair', 'Cloudy', 'Overcast', 'Fog', 'Freezing Fog', 'Light Rain', 'Rain', 'Heavy Rain',
    'Freezing Rain', 'Heavy Freezing Rain', 'Sleet', 'Heavy Sleet', 'Light Snowfall', 'Snowfall', 'Heavy Snowfall',
    'Rain Shower', 'Heavy Rain Shower', 'Sleet Shower', 'Heavy Sleet Shower', 'Snow Shower', 'Heavy Snow Shower',
    'Lightning', 'Hail', 'Thunderstorm', 'Heavy Thunderstorm', 'Storm'
]

fill_na_with_zeros: Final[SimpleImputer] = SimpleImputer(missing_values=pd.NA, strategy='constant', fill_value=0)
float_to_uint8: Final[FunctionTransformer] = FunctionTransformer(lambda x: x.astype(np.uint8), feature_names_out='one-to-one')
coco_one_hot_encoder: Final[OneHotEncoder] = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='error', dtype=np.uint8)

preprocessing_steps: Final[Pipeline] = Pipeline(
    steps=[
        ('fill_na_with_zeros', fill_na_with_zeros),
        ('coco_float_to_uint8', ColumnTransformer(
            transformers=[
                ('float_to_uint8', float_to_uint8, ['coco']),
            ],
            remainder='passthrough'
        )),
        ('coco_one_hot_encoder', ColumnTransformer(
            transformers=[
                ('one_hot_encoder', coco_one_hot_encoder, ['float_to_uint8__coco']),
            ],
            remainder='passthrough',
        )),
    ],
    verbose=True
).set_output(transform='pandas')


def preprocess_weather_data(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the weather data.

    The preprocessing involves the following steps:
    - Fill missing values with zeros
    - Convert the 'coco' column from float to uint8
    - One hot encode the 'coco' column

    Since scikit-learn automatically adds the name of the last step to the column names, we rename the columns to remove the step names at the end as well.
    We also rename the 'coco' columns to include the actual weather conditions corresponding to the codes.

    :param weather_df: the weather data with 'timestamp', 'temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun', and  'coco' columns.
    :return: the preprocessed weather data with 'timestamp', 'f_temp', 'f_dwpt', 'f_rhum', 'f_prcp', 'f_snow', 'f_wdir', 'f_wspd', 'f_wpgt',
        'f_pres', and 'f_tsun' columns in addition to the one hot encoded 'f_coco' columns.
    """
    weather_df: pd.DataFrame = preprocessing_steps.fit_transform(weather_df)

    # Rename the columns
    weather_df.rename(columns={feature: f"f_{feature.split('__')[-1]}" for feature in weather_df.columns}, inplace=True)
    weather_df.rename(
        columns={f"f_coco_{i}": f"f_coco_{WEATHER_CONDITION_CODES[i]}" for i in range(len(WEATHER_CONDITION_CODES))},
        inplace=True, errors='ignore'
    )
    return weather_df
