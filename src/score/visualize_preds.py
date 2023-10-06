import pandas as pd
import json
import plotly.graph_objects as go
from src.get_processed_data import get_processed_data
from src.configs.load_config import ConfigLoader
import warnings


def plot_preds_on_series(preds: pd.DataFrame, data: pd.DataFrame, events_path: str = 'data/raw/train_events.csv',
                         features_to_plot: list = None, number_of_series_to_plot: int = 1):
    """ This function plots the predictions on the series data. It also plots the real events on the series data."""
    # get the unique series ids from the preds
    # the predictions have the series ids as strings
    # so we encode them to ints later
    series_ids = preds['series_id'].unique()

    # Load the encoding JSON
    with open('series_id_encoding.json', 'r') as f:
        id_encoding = json.load(f)

    # loop through the series ids
    for id in series_ids[:number_of_series_to_plot]:
        # Encode the series ID
        id = id_encoding.get(str(id))

        if id is not None:
            current_series = data[data['series_id'] == id]
            real_events = pd.read_csv(events_path)
            # apply id encoding to events
            real_events['series_id'] = real_events['series_id'].map(id_encoding)
            # take the events for the current series
            current_events = real_events.loc[real_events['series_id'] == id]
            # apply id encoding to preds
            preds['series_id'] = preds['series_id'].map(id_encoding)
            # take the preds for the current series
            current_preds = preds.loc[preds['series_id'] == id]
            # separate the series by awake values to plot with different colors
            awake_0_df = current_series[current_series['awake'] == 0]
            awake_1_df = current_series[current_series['awake'] == 1]
            awake_2_df = current_series[current_series['awake'] == 2]

            # the steps of the dataframes awake-0_df will have diff > 1 for when a jump occurs
            # we need to find the indices of the jumps and use the indices to give them a group number
            # then using .groupby('group') we can plot the groups separately
            # this way the jumps will not be connected

            # Writing this part in place was going to be too hard so i didnt do it
            # this way works but gives warnings about SettingWithCopyWarning
            # so we just ignore those warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # get the diff of the steps
                awake_0_df['diff'] = awake_0_df['step'].diff()
                # get the indices of the jumps
                jump_indices = awake_0_df[awake_0_df['diff'] > 1].index
                # create a new column with the group number
                awake_0_df['group'] = 0
                # loop through the jump indices and increment the group number
                for i in jump_indices:
                    awake_0_df.loc[i:, 'group'] += 1

                awake_1_df['diff'] = awake_1_df['step'].diff()
                jump_indices = awake_1_df[awake_1_df['diff'] > 1].index
                awake_1_df['group'] = 0
                for i in jump_indices:
                    awake_1_df.loc[i:, 'group'] += 1

                awake_2_df['diff'] = awake_2_df['step'].diff()
                jump_indices = awake_2_df[awake_2_df['diff'] > 1].index
                awake_2_df['group'] = 0
                for i in jump_indices:
                    awake_2_df.loc[i:, 'group'] += 1

            # create the figure
            fig = go.Figure()
            # if features_to_plot is None, plot all the features
            if features_to_plot is None:
                features_to_plot = data.columns.values
            for feature_to_plot in features_to_plot:
                # some features are not meant to be plotted like step, series_id, awake, timestamp
                if feature_to_plot != 'step' and feature_to_plot != 'series_id' and feature_to_plot != 'awake' and feature_to_plot != 'timestamp':
                    # This part plots the features for awake=0, awake=1, awake=2 in a way that the jumps are not connected
                    # and we get only 1 legend item for the entire group by showing legend only on the first trace
                    awake_0_df.groupby('group').apply(lambda x: fig.add_trace(go.Scatter(x=x['step'],
                                                                                         y=x[feature_to_plot], mode='lines', name=feature_to_plot+'Awake=0',
                                                                                         line=dict(color='blue'), legendgroup=feature_to_plot+'Awake=0',
                                                                                         showlegend=True if x.name == 0 else False)))
                    awake_1_df.groupby('group').apply(lambda x: fig.add_trace(go.Scatter(x=x['step'],
                                                                                         y=x[feature_to_plot], mode='lines', name=feature_to_plot+'Awake=1',
                                                                                         line=dict(color='red'), legendgroup=feature_to_plot+'Awake=1',
                                                                                         showlegend=True if x.name == 0 else False)))
                    awake_2_df.groupby('group').apply(lambda x: fig.add_trace(go.Scatter(x=x['step'],
                                                                                         y=x[feature_to_plot], mode='lines', name=feature_to_plot+'Awake=2',
                                                                                         line=dict(color='green'), legendgroup=feature_to_plot+'Awake=2',
                                                                                         showlegend=True if x.name == 0 else False)))
                    fig.update_xaxes(title='Timestamp')
                    fig.update_yaxes(title='Feature values')
                    fig.update_layout(title=f'Anglez for Series ID: {id}')
                    fig.update_xaxes(tickvals=current_series['step'][::len(current_series) // 10], tickangle=45)

            # before showing the figure make vertical lines for the current_events and current_preds
            # the vertical lines have annotations that show the real and predicted values and the timestamps
            # and the annotations are colored according to the event type
            for current_onset in current_events[current_events['event'] == 'onset']['step'].dropna():
                fig.add_vline(x=current_onset, line_dash="dash", line_color="black", line_width=2,
                              annotation_text=f'<span style="color:black">real_onset:<br> {current_onset}</span>',
                              annotation_position="top", name=f'Vertical Line at x={current_onset}', annotation_textangle=315,
                              )
            for current_wakeup in current_events[current_events['event'] == 'wakeup']['step'].dropna():
                fig.add_vline(x=current_wakeup, line_dash="dash", line_color="green", name="wakeup", line_width=2,
                              annotation_text=f'<span style="color:green">real_wakeup:<br> {current_wakeup}</span>',
                              annotation_position="top", annotation_textangle=315)

            for current_pred_onset in current_preds[current_preds['event'] == 'onset']['step'].dropna():
                fig.add_vline(x=current_pred_onset, line_dash="dash", line_color="red", line_width=2,
                              annotation_text=f'<span style="color:red">pred_onset:<br> {current_pred_onset}</span>',
                              annotation_position="top", name=f'Vertical Line at x={current_pred_onset}', annotation_textangle=315,
                              )
            for current_pred_wakeup in current_preds[current_preds['event'] == 'wakeup']['step'].dropna():
                fig.add_vline(x=current_pred_wakeup, line_dash="dash", line_color="orange", name="wakeup", line_width=2,
                              annotation_text=f'<span style="color:orange"> pred_wakeup:<br> {current_pred_wakeup}</span>',
                              annotation_position="top", annotation_textangle=315,
                              )
            # the normal margin doesnt work for the annotations
            fig.update_layout(margin=dict(t=160))
            # show the figure
            fig.show()


if __name__ == "__main__":
    # for testing you need a submission.csv file in the main folder
    # and it read the processed data
    preds = pd.read_csv("submission.csv")
    config = ConfigLoader("test/test_config.json")
    series_path = 'data/raw/train_series.parquet'
    featured_data = get_processed_data(config, series_path, save_output=True)
    # plot the predictions on the series data
    plot_preds_on_series(preds, featured_data)
