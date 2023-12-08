from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..preprocessing.pp import PP


@dataclass
class AddSegmentationLabels(PP):
    """Preprocessing step that adds the segmentation labels to the data
    """

    def preprocess(self, data: dict) -> dict:
        """Adds the segmentation labels to the data.

        It will add 3 columns which is a result from the one-hot encoding of the 'awake' column.

        :param data: The dataframe to add the segmentation labels to
        :return: The dataframe with the segmentation labels
        """
        # Apply one-hot encoding using dummies to the 'awake' column and call then hot-asleep, hot-awake and hot-NaN as type int8
        # TODO Check if the awake column is present #190
        for sid in data.keys():

            awake = data[sid]['awake']

            data[sid] = pd.get_dummies(data[sid], columns=['awake'], prefix='hot', dtype=np.int8)
            name_map = {'hot_0': 'hot-asleep', 'hot_1': 'hot-awake', 'hot_2': 'hot-NaN', 'hot_3': 'hot-unlabeled'}
            data[sid].rename(columns=name_map, inplace=True)

            for name in name_map.values():
                if name not in data[sid].columns:
                    data[sid][name] = 0

            data[sid]['awake'] = awake.astype(np.int8)

            pad_type = {'step': np.int32, 'enmo': np.float32,
                        'anglez': np.float32, 'timestamp': 'datetime64[ns]'}
            data[sid] = data[sid].astype(pad_type)
        return data
