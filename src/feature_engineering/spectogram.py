import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from matplotlib.backends.backend_agg import FigureCanvasAgg


#stft(enmo.to_numpy(), fs, nperseg=256)
def spectogram(plot = False, y = None, fs = None, nperseg = 256):
    # given the time series data with the sampling frequency calculates the
    # short-time-fourier-transform using the scipy function stft
    # by default stft doesnt return an image so this function returns the spectogram image
    # nperseg is a variable that determines the resolution in the y and x axes
    # if you increase it the resolution you get on the y axis increases
    # but it also decresases the resolution in the x axis
    if fs is None:
        raise ValueError("Sampling freq. must be specified")
    if not isinstance(y, np.ndarray):
        raise ValueError("y must be a numpy.ndarray")
    f, t, Zxx = signal.stft(y, fs, nperseg=nperseg)
    # Zxx is an array of complex number
    # spectograms plot the magnitude
    # so we take the abs value first
    Y_abs = np.abs(Zxx)
    fig, ax = plt.subplots()
    # there can be huge outliers which mess up how the plot works so the upper limit is the mean + 5*std
    # for displaying the color axis
    quadmesh = ax.pcolormesh(t, f, Y_abs, vmax=np.mean(Y_abs)+5*np.std(Y_abs), shading='gouraud')
    # quadmesh is a maplotlib Quadmesh object and not an array
    # not a high priority rn so im leaving this for later
    # TODO figure out how to get rgb array from quadmesh witout rendering the image
    # we already have the x and y axis ve need to project the z axis as colors on to the grid
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
    plt.close(fig)

    return image_array


