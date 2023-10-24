import plotly.graph_objects as go
import pandas as pd


def add_pred_vlines(fig: go.Figure, current_events: pd.DataFrame, current_preds: pd.DataFrame) -> go.Figure:
    """
    This function takes in the current figure and events and draws vertical lines for each event in the figure.

    Args:
        fig: The figure to add the vertical lines to
        events: The events to draw vertical lines for
        preds: The predictions to draw vertical lines for
    """
    # We cant vectorize this in plotly so we loop over the events
    for current_onset in current_events[current_events['event'] == 'onset']['step'].dropna():
        fig.add_vline(x=current_onset, line_dash="dash", line_color="black", line_width=2,
                      annotation_text=f'<span style="color:black">real_onset:<br> {current_onset}</span>',
                      annotation_position="top", name=f'Vertical Line at x={current_onset}', annotation_textangle=315)

    for current_wakeup in current_events[current_events['event'] == 'wakeup']['step'].dropna():
        fig.add_vline(x=current_wakeup, line_dash="dash", line_color="green", line_width=2,
                      annotation_text=f'<span style="color:green">real_wakeup:<br> {current_wakeup}</span>',
                      annotation_position="top", annotation_textangle=315)

    for current_pred_onset in current_preds[current_preds['event'] == 'onset']['step'].dropna():
        fig.add_vline(x=current_pred_onset, line_dash="dash", line_color="red", line_width=2,
                      annotation_text=f'<span style="color:red">pred_onset:<br> {current_pred_onset}</span>',
                      annotation_position="top", name=f'Vertical Line at x={current_pred_onset}', annotation_textangle=315)

    for current_pred_wakeup in current_preds[current_preds['event'] == 'wakeup']['step'].dropna():
        fig.add_vline(x=current_pred_wakeup, line_dash="dash", line_color="orange", line_width=2,
                      annotation_text=f'<span style="color:orange"> pred_wakeup:<br> {current_pred_wakeup}</span>',
                      annotation_position="top", annotation_textangle=315)
