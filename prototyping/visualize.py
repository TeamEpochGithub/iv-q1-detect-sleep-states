import json
import sys

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from prototyping.features import add_features
from prototyping.model import TimeSeriesSegmentationModel

window_length = 17280


def masked_x(index: pd.Index, mask):
    """where the mask is false, replace the value with NaN, returns a copy"""
    x = index.to_numpy(copy=True, dtype=np.float32)
    x[~mask] = np.nan
    return x


def plot_data(series, segment_pred, event_pred, title_id, feature, subsample=1, plot_limit=None, save=None):
    series = series[:plot_limit:subsample]

    # segment pred has shape (windows,1,1440)
    # concatenate and upsample by 12x to (windows*12780)
    segment_pred = np.concatenate(segment_pred, axis=1)[0]
    segment_pred = np.repeat(segment_pred, 12, axis=0)

    event_pred = np.transpose(event_pred, (1, 0, 2))
    onset_pred = np.concatenate(event_pred[0], axis=0)
    onset_pred = np.repeat(onset_pred, 12, axis=0)
    awake_pred = np.concatenate(event_pred[1], axis=0)
    awake_pred = np.repeat(awake_pred, 12, axis=0)

    segment_pred = segment_pred[:plot_limit:subsample]
    onset_pred = onset_pred[:plot_limit:subsample]
    awake_pred = awake_pred[:plot_limit:subsample]

    awake_0_x = masked_x(series.index, series['awake'] == 0)
    awake_1_x = masked_x(series.index, series['awake'] == 1)
    awake_2_x = masked_x(series.index, series['awake'] == 2)
    awake_3_x = masked_x(series.index, series['awake'] == 3)

    # normalize the feature
    series[feature] = (series[feature] - series[feature].mean()) / series[feature].std()

    fig_anglez = px.line()
    fig_anglez.add_scatter(x=awake_0_x, y=series[feature], mode='lines', name='0 (Sleep)', line=dict(color='blue'))
    fig_anglez.add_scatter(x=awake_1_x, y=series[feature], mode='lines', name='1 (Awake)', line=dict(color='red'))
    fig_anglez.add_scatter(x=awake_2_x, y=series[feature], mode='lines', name='2 (NaN)', line=dict(color='green'))
    fig_anglez.add_scatter(x=awake_3_x, y=series[feature], mode='lines', name='3 (Unlabeled)', line=dict(color='grey'))

    fig_anglez.add_scatter(x=series.index, y=segment_pred, mode='lines', name='segmentation', line=dict(color='light grey'))
    fig_anglez.add_scatter(x=series.index, y=onset_pred, mode='lines', name='onset', line=dict(color='orange'))
    fig_anglez.add_scatter(x=series.index, y=awake_pred, mode='lines', name='awake', line=dict(color='black'))
    fig_anglez.update_xaxes(title='Timestamp')
    fig_anglez.update_yaxes(title=feature)
    fig_anglez.update_layout(title=f'Series {title_id} - {feature} - Subsampled {subsample}x')

    # Show or save the plot
    if save is None:
        fig_anglez.show()
    elif save == 'html':
        plotly.offline.plot(fig_anglez, filename=f'./plots/pred/{title_id}-{subsample}x.html', auto_open=False)
    else:
        fig_anglez.write_image(f'./plots/pred/{title_id}-{subsample}x.{save}', width=2000, height=600)


def series_to_model_input(df):
    # Group by 'window' column
    windows = df.groupby('window')
    processed_data = []
    for window_id, window_data in windows:
        if len(window_data) != window_length:
            continue

        # Drop 'window' column and add features
        window_data = window_data.drop(columns=['window'])
        featured = add_features(window_data)
        processed_data.append(np.array(featured).T)

    # Convert to a numpy array
    processed_data = np.array(processed_data)

    scaler = StandardScaler()
    reshaped = processed_data.transpose(0, 2, 1).reshape(-1, processed_data.shape[1])
    scaler.fit(reshaped)

    for window in tqdm(processed_data, desc='Transforming data'):
        window[:] = scaler.transform(window.T).T

    return processed_data


if __name__ == '__main__':
    # Load the pandas version of the series
    first_timestamps = json.load(open('./data/processed/train/first_timestamps.json'))
    pbar = tqdm(first_timestamps, file=sys.stdout)
    series = dict()
    series_ids = list()
    for series_id in pbar:
        data = pd.read_parquet(f'./data/processed/train/labeled/{series_id}.parquet')
        series[series_id] = data
        series_ids.append(series_id)

    # load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesSegmentationModel().double().to(device)
    model.load_state_dict(torch.load('./tm/model_9.pt'))
    print('Loaded model')

    # make predictions
    for series_id in series_ids[:10]:
        df = series[series_id]
        df = df.groupby('window').filter(lambda x: len(x) == window_length)
        model_input = series_to_model_input(df)
        model_input = torch.from_numpy(model_input).to(device)

        if model_input.shape[0] == 0:
            continue

        print(f'Predicting series {series_id}')
        segment_pred, event_pred = model(model_input)
        segment_pred = segment_pred.cpu().detach().numpy()
        event_pred = event_pred.cpu().detach().numpy()

        print(f'Plotting series {series_id}')
        # plot the predictions
        plot_data(df, segment_pred, event_pred, series_id, 'anglez', subsample=1,
                  save='html')
