import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import sys

from wavwrapper import wavfile, monowrapper
from windowing import overlapped_window

# Attempt to load progressbar2 library. If it is not available, declare
# a stub ProgressBar class that does nothing.
try:
    from progressbar import ProgressBar
except:
    class ProgressBar (object):
        def __init__(*args, **kwargs):
            pass
        def update(self, x):
            pass
        def finish(self):
            pass

def main(input_file=None, window_size="1024", scale="log"):

    # Process command-line args.
    if input_file is None:
        sys.stderr.write("usage: python %s <input_file.wav> [window_size=int, default 1024] [scale=log|linear, default log]\n" % sys.argv[0])
        return 1
        
    window_size = int(window_size)
    
    if not scale in ['log', 'linear']:
        sys.stderr.write("error: '%s' is not a valid scale, choose 'log' or 'linear'.\n" % scale)
        return 1

    # Open wave file and load frame rate, number of channels, sample width, and number of frames.
    w = wavfile(input_file)
    
    # Catch case where there are more than 2 channels.
    if w.get_param('nchannels') > 2:
        sys.stderr.write("error: only mono and stereo tracks are supported\n")
        return 1
    
    # Catch case where there is less than one window of audio.
    if w.get_param('nframes') < window_size:
        sys.stderr.write("error: audio file is shorter than configured window size\n")
        return 1
    
    # Hann window function coefficients.
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(window_size)) / window_size)
    
    # Hann window must have 4x overlap for good results.
    overlap = 4
    
    # Y will hold the DFT of each window. We use acc and bar for displaying progress.
    Y = []
    acc = 0
    bar = ProgressBar(max_value=w.get_param('nframes') * overlap)
    
    # Process each window of audio.
    for x in overlapped_window(monowrapper(w), window_size, overlap):
        y = np.fft.rfft(x * hann)[:window_size//2]
        Y.append(y)
        acc += window_size
        bar.update(acc)
    
    # Inform progress bar that the computation is complete.
    bar.finish()
    
    # Normalize data and convert to dB.
    Y = np.column_stack(Y)
    Y = np.absolute(Y) * 2.0 / np.sum(hann)
    Y = Y / np.power(2.0, (8 * w.get_param('sampwidth') - 1))
    Y = (20.0 * np.log10(Y)).clip(-120)

    # Time domain: We have Y.shape[1] windows, so convert to seconds by multiplying
    # by window size, dividing by sample rate, and dividing by the overlap rate.
    t = np.arange(0, Y.shape[1], dtype=np.float) * window_size / w.get_param('framerate') / overlap
    
    # Frequency domain: There are window_size/2 frequencies represented, and we scale
    # by dividing by window size and multiplying by sample frequency.
    f = np.arange(0, window_size / 2, dtype=np.float) * w.get_param('framerate') / window_size
    
    # Plot the spectrogram.
    ax = plt.subplot(111)
    plt.pcolormesh(t, f, Y, vmin=-120, vmax=0)
    
    # Use log scale above 100 Hz, linear below.
    if scale == 'log':
        plt.yscale('symlog', linthreshy=100, linscaley=0.25)
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    # Set x/y limits by using the maximums from the time/frequency arrays.
    plt.xlim(0, t[-1])
    plt.ylim(0, f[-1])
    
    # Set axis labels.
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    
    # Show legend and set label.
    cbar = plt.colorbar()
    cbar.set_label("Intensity (dB)")
    
    # Display spectrogram.
    plt.show()
    
    return 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
