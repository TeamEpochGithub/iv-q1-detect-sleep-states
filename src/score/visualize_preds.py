import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from src.get_processed_data import get_processed_data
from src.configs.load_config import ConfigLoader


def plot_preds_on_series(preds: pd.DataFrame, data: pd.DataFrame, events_path: str = 'data/raw/train_events.csv', features_to_plot: list = None):
    '''The data is the featured adat we use at the end of the pipeline'''
    # for now the train events is hard coded
    # events = pd.read_csv('data/raw/train_events.csv')
    # get the unique series ids from the preds
    series_ids = preds['series_id'].unique()

    # Load the encoding JSON
    with open('series_id_encoding.json', 'r') as f:
        id_encoding = json.load(f)

    # Create an empty figure
    fig = px.line()

    # loop through the series ids
    for id in series_ids[:1]:
        # Encode the series ID
        id = id_encoding.get(str(id))

        if id is not None:
            current_series = data[data['series_id'] == id]
            real_events = pd.read_csv(events_path)
            # apply id encoding to events
            real_events['series_id'] = real_events['series_id'].map(id_encoding)
            current_events = real_events.loc[real_events['series_id'] == id]
            preds['series_id'] = preds['series_id'].map(id_encoding)
            current_preds = preds.loc[preds['series_id'] == id]
            awake_0_df = current_series[current_series['awake'] == 0]
            awake_1_df = current_series[current_series['awake'] == 1]
            awake_2_df = current_series[current_series['awake'] == 2]

            # the steps of the dataframes awake-0_df will have diff > 1 for when a jump occurs
            # we need to find the indices of the jumps and use the indices to give them a group number
            # then using .groupby('group') we can plot the groups separately

            # get the diff of the steps
            awake_0_df['diff'] = awake_0_df['step'].diff()
            # get the indices of the jumps
            jump_indices = awake_0_df[awake_0_df['diff'] > 1].index
            # create a new column with the group number
            awake_0_df['group'] = 0
            # loop through the jump indices and increment the group number
            for i in jump_indices:
                awake_0_df.loc[i:, 'group'] += 1
            # current_series = current_series[binary_mask == 1]
            fig = px.line()
            for feature_to_plot in ['anglez', 'enmo']:
                # Create the plots for the current feature
                awake_0_df.groupby('group').apply(lambda x: fig.add_scatter(x=x['step'], y=x[feature_to_plot], mode='lines', name=feature_to_plot+'Awake=0', line=dict(color='blue')))
                # fig.add_scatter(x=awake_0_df['step'], y=awake_0_df[feature_to_plot], mode='lines', name=feature_to_plot+'Awake=0', line=dict(color='blue'))
                fig.add_scatter(x=awake_1_df['step'], y=awake_1_df[feature_to_plot], mode='lines', name=feature_to_plot+'Awake=1', line=dict(color='red'))
                fig.add_scatter(x=awake_2_df['step'], y=awake_2_df[feature_to_plot], mode='lines', name=feature_to_plot+'Awake=2', line=dict(color='green'))
                fig.update_xaxes(title='Timestamp')
                fig.update_yaxes(title='Feature values')
                fig.update_layout(title=f'Anglez for Series ID: {id}')
                fig.update_xaxes(tickvals=current_series['step'][::len(current_series) // 10], tickangle=45)
                # Show the first plot
            # before showing the figure make vertical lines for the current_events and current_preds
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
            fig.update_traces(connectgaps=False)
            fig.update_layout(margin=dict(t=160), showlegend=True)
            fig.show()


if __name__ == "__main__":
    # hard coded for now
    preds = pd.read_csv("C:/Users/Tolga/Downloads/submission.csv")
    config = ConfigLoader("test/test_config.json")
    series_path = 'data/raw/train_series.parquet'
    featured_data = get_processed_data(config, series_path, save_output=True)
    plot_preds_on_series(preds, featured_data)
