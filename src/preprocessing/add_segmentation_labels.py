import numpy as np
import pandas as pd

from src.preprocessing.pp import PP


class AddSegmentationLabels(PP):
    """Preprocessing step that adds the segmentation labels to the data
    """

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adds the segmentation labels to the data. It will add 3 columns which is a result from the one-hot encoding of the 'awake' column.
        :param data: The dataframe to add the segmentation labels to
        :return: The dataframe with the segmentation labels
        """
        # Apply one-hot encoding using dummies to the 'awake' column and call then hot-asleep, hot-awake and hot-NaN as type int8

        awake = data['awake']

        data = pd.get_dummies(data, columns=['awake'], prefix='hot', dtype=np.int8)
        name_map = {'hot_0': 'hot-asleep', 'hot_1': 'hot-awake', 'hot_2': 'hot-NaN'}
        data.rename(columns=name_map, inplace=True)

        for name in name_map.values():
            if name not in data.columns:
                data[name] = 0

        data['awake'] = awake.astype(np.int8)

        pad_type = {'step': np.uint32, 'series_id': np.uint16, 'enmo': np.float32,
                    'anglez': np.float32, 'timestamp': 'datetime64[ns]'}
        data = data.astype(pad_type)
        return data
