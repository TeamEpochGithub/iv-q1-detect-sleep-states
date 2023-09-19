import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(1,'../src')
sys.path.insert(1,'../data')
# pycharm gives error but it works
from scoring import score

column_names = {
'series_id_column_name': 'series_id',
'time_column_name': 'time',
'event_column_name': 'event',
'score_column_name': 'score',
}

tolerances = {'pass': [1.0]}
solution = pd.DataFrame({
'series_id': ['a', 'a', 'a', 'a'],
'event': ['start', 'pass', 'pass', 'end'],
'time': [0, 10, 20, 30],
})
submission = pd.DataFrame({
'series_id': ['a', 'a', 'a'],
'event': ['pass', 'pass', 'pass'],
'score': [1.0, 1.0, 1.0],
'time': [10, 20, 40],
})

train_events = pd.read_csv("../data/train_events.csv")
# pick an event to look at
event_id = train_events['series_id'].unique()[0]
train_event = train_events.loc[train_events['series_id'] == event_id]
# now only take onsets from this event also dropping the nans (nans lead to lower score)
onset_steps = train_event.loc[train_event['event'] == 'onset']['step'].dropna()
onset_steps = onset_steps.to_numpy()
#print(onset_steps)

# now using these steps create a sample submission

#print(test_score)

def gaussian_predictions(prediction_timestamps, prediction_confidences, sigmas, num_points=11):
    # this function will, for a given array of timesteps and their confidences, will generate a
    # gaussian curve around each prediction using the given sigma value
    # and it will generate new predcitions each 1 sigma apart up to 5 sigma in both directions of the original prediction
    # so for each predcition it will return the original predicition and num_points-1 more predictions around it

    # Create an array of x values spanning from -5 sigma to +5 sigma
    all_results = np.empty((0, 2))
    # if num_points is even it destroys the original number so we make it an odd number
    if num_points % 2 == 0:
        num_points += 1
    for i in range(len(prediction_timestamps)):
        timestamp = prediction_timestamps[i]
        confidence = prediction_confidences[i]
        sigma = sigmas[i]
        x_values = np.linspace(timestamp - 5 * sigma, timestamp + 5 * sigma, num_points)

        # Calculate the corresponding y values using the Gaussian formula
        y_values = confidence * np.exp(-(x_values - timestamp) ** 2 / (2 * sigma ** 2))

        # Combine x and y values into a list of (x, y) tuples
        results = np.column_stack((x_values, y_values))
        all_results = np.concatenate((all_results, results), axis=0)

    # returns a 2 column vector where the 0th col is the timestamps and the 1st column is the y values
    # to plot do this: plt.plot(new_points[:,0], new_points[:,1])
    # timestamps = new_points[:,0]
    # confidences = new_points[:,1]
    return all_results

test_timestamps = onset_steps
test_confidences = np.linspace(0.4,1,len(onset_steps))
test_sigmas = [1] * len(onset_steps)

old_points = np.array([test_timestamps, test_confidences])
new_points = gaussian_predictions(prediction_timestamps=test_timestamps, prediction_confidences=test_confidences,\
                                  sigmas=test_sigmas, num_points=29)

#new points has an x column and a y column

# offset the time of the artificial predicitons
new_points[:,0] = np.add(new_points[:,0], 10000)
plt.figure()
plt.ylim((0,1))
plt.scatter(old_points[0], old_points[1])
plt.plot(new_points[:,0], new_points[:,1], c='r')
plt.show()

# now using the newly generated points see if the recall values keep increasing as confidences go down

tolerances = {'onset': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]}
solution = pd.DataFrame({
'series_id': [event_id] * len(onset_steps),
'event': ['onset'] * len(onset_steps),
'time': onset_steps,
})

# for now keeping it the same as the answer
submission = pd.DataFrame({
'series_id': [event_id] * len(new_points[:,0]),
'event': ['onset'] * len(new_points[:,0]),
'time': new_points[:,0],
'score': new_points[:,1]
})

test_score = score(solution, submission, tolerances, **column_names, use_scoring_intervals=False, plot_precision_recall=False)

print(test_score)