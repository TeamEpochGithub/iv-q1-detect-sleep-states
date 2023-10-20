import plotly.graph_objects as go
from util.add_pred_vlines import add_pred_vlines
import numpy as np


def plot_w_plotly(current_series, current_events, current_preds, id_decoding, id, features_to_plot):
    fig = go.Figure()
    for feature_to_plot in features_to_plot:
        # Some features are not meant to be plotted like step, series_id, awake, timestamp
        if feature_to_plot == 'anglez' or feature_to_plot == 'enmo' or 'f_' in feature_to_plot:
            mask = current_series['awake'] == 0
            x = current_series['step'].to_numpy(copy=True, dtype=np.float32)
            x[~mask] = np.nan
            fig.add_trace(go.Scatter(x=x, y=current_series[feature_to_plot].values,
                                     mode='lines', name=feature_to_plot+'Awake=0',
                                     line=dict(color='blue'), legendgroup=feature_to_plot+'Awake=0',
                                     showlegend=True))
            mask = current_series['awake'] == 1
            x = current_series['step'].to_numpy(copy=True, dtype=np.float32)
            x[~mask] = np.nan
            fig.add_trace(go.Scatter(x=x, y=current_series[feature_to_plot].values,
                                     mode='lines', name=feature_to_plot+'Awake=1',
                                     line=dict(color='red'), legendgroup=feature_to_plot+'Awake=1',
                                     showlegend=True))
            mask = current_series['awake'] == 2
            x = current_series['step'].to_numpy(copy=True, dtype=np.float32)
            x[~mask] = np.nan
            fig.add_trace(go.Scatter(x=x, y=current_series[feature_to_plot].values,
                                     mode='lines', name=feature_to_plot+'Awake=2',
                                     line=dict(color='green'), legendgroup=feature_to_plot+'Awake=2',
                                     showlegend=True))
            fig.update_xaxes(title='Timestamp')
            fig.update_yaxes(title='Feature values')
            fig.update_layout(
                title=f'Anglez for Series ID: {id_decoding[id]}-{id}')
            fig.update_xaxes(tickvals=current_series['step'][::len(
                current_series) // 10], tickangle=45)

    # Before showing the figure make vertical lines for the current_events and current_preds
    # the vertical lines have annotations that show the real and predicted values and the timestamps
    # and the annotations are colored according to the event type
    add_pred_vlines(fig, current_events, current_preds)
    # The normal margin doesnt work for the annotations
    fig.update_layout(margin=dict(t=160))
    fig.show()
