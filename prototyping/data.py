import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import joblib
from prototyping.features import add_features

# Constants
window_length = 17280
reduce_factor = 12

split = 0.7

class TimeSeriesDataset(Dataset):
    def __init__(self, data, segmentation_labels, event_classification_labels):
        self.data = data
        self.segmentation_labels = segmentation_labels
        self.event_classification_labels = event_classification_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx],
            'segmentation_label': self.segmentation_labels[idx],
            'event_classification_label': self.event_classification_labels[idx]
        }
        return sample


def get_data_loader() -> DataLoader:

    # load the arrays from disk
    processed_data = np.load('./data/processed/train/featured/processed_data.npy')
    segmentation_labels = np.load('./data/processed/train/featured/segmentation_labels.npy')
    event_classification_labels = np.load('./data/processed/train/featured/event_classification_labels.npy')

    batch_size = 32  # Adjust as needed
    dataset = TimeSeriesDataset(processed_data, segmentation_labels, event_classification_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


if __name__ == '__main__':

    first_timestamps = json.load(open(f'./data/processed/train/first_timestamps.json'))

    # select the first split of series ids from the keys in first_timestamps
    all_sids = list(first_timestamps.keys())
    all_sids.sort()
    train_ids = all_sids[:int(len(all_sids)*split)]

    series = {}
    for series_id in tqdm(train_ids, desc='Loading data'):
        series[series_id] = pd.read_parquet(f'./data/processed/train/labeled/{series_id}.parquet')

    # Create a list to store processed data
    processed_data = []
    segmentation_labels = []
    event_classification_labels = []

    for series_id, df in tqdm(series.items(), desc='Processing data'):
        # Group by 'window' column
        windows = df.groupby('window')

        for window_id, window_data in windows:
            if len(window_data) != window_length:
                continue  # Skip windows that are not 17280 long
            if window_data['awake'].max() > 1:
                continue  # Skip windows that include NaN or unlabeled

            # Process 'awake' column for segmentation
            segmentation_label = window_data['awake'].values.reshape(-1, reduce_factor)
            segmentation_label = np.median(segmentation_label, axis=1)  # Median per 12 steps

            # Process 'awake' column for event classification
            awake_diff = np.diff(window_data['awake'], append=1)
            event_label = np.zeros((len(awake_diff), 2))
            event_label[awake_diff == -1, 0] = 1
            event_label[awake_diff == 1, 1] = 1
            event_label = event_label.reshape(-1, reduce_factor, 2)
            event_label = np.max(event_label, axis=1)  # Max per 12 steps

            # Drop 'window' column and add features
            window_data = window_data.drop(columns=['window'])
            featured = add_features(window_data)

            processed_data.append(np.array(featured).T)
            segmentation_labels.append(segmentation_label)
            event_classification_labels.append(event_label.T)

    # Convert to a numpy array
    processed_data = np.array(processed_data)
    segmentation_labels = np.array(segmentation_labels)
    event_classification_labels = np.array(event_classification_labels)

    # Standardize the data
    print('Fitting scaler')
    scaler = StandardScaler()

    # processed data has shape (num_windows, num_features, window_length)
    # scaler requires shape (num_windows * window_length, num_features)
    reshaped = processed_data.transpose(0, 2, 1).reshape(-1, processed_data.shape[1])
    scaler.fit(reshaped)

    for window in tqdm(processed_data, desc='Transforming data'):
        window[:] = scaler.transform(window.T).T

    # save the scalar
    joblib.dump(scaler, './tm/std_scaler.bin', compress=True)

    # save the numpy arrays to disk
    np.save('./data/processed/train/featured/processed_data.npy', processed_data)
    np.save('./data/processed/train/featured/segmentation_labels.npy', segmentation_labels)
    np.save('./data/processed/train/featured/event_classification_labels.npy', event_classification_labels)

