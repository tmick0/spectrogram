import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import wave
import sys

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

def main(input_file=None, window_size="1024"):

    # Process command-line args.
    if input_file is None:
        sys.stderr.write("usage: python %s <input_file.wav> [window_size=1024]\n" % sys.argv[0])
        return 1
    window_size = int(window_size)

    # Open wave file and load frame rate, number of channels, sample width, and number of frames.
    w = wave.open(input_file, 'r')
    sample_freq = w.getframerate()
    chans = w.getnchannels()
    width = w.getsampwidth()
    frame = w.getnframes()
    
    # Use an 8-bit integer for single-byte samples, 16-bit integer for 2-byte samples.
    if width == 1:
        dtype = np.int8
    elif width == 2:
        dtype = np.int16
    else:
        sys.stderr.write("error: only 8-bit and 16-bit signed samples are supported\n")
        return 1
    
    # Hann window function coefficients.
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(window_size)) / window_size)
    
    # Hann window must have 4x overlap for good results.
    overlap = 4
    
    # Y will hold the DFT of each window. We use acc and bar for displaying progress.
    Y = []
    acc = 0
    bar = ProgressBar(max_value=frame*overlap)
    
    # X will hold residual window content for overlapping.
    X = []
    
    # Process each window of audio.
    while True:
        
        # If this is the first read, load an entire window, otherwise just 1/overlap of a window.
        read_size = window_size if len(X) == 0 else window_size / overlap
        
        # Load raw audio data into a numpy array.
        x = w.readframes(read_size)
        x = np.fromstring(x, dtype=dtype)
        
        # If the window read was short, end.
        if len(x) != read_size * chans:
            break
        
        # Reshape our array into an N*c matrix in order to separate the channels.
        x = np.reshape(x, (read_size, chans))

        # Average the channels if the audio is stereo.
        if chans > 1:
            x = (x[:,0] + x[:,1]) / 2
        else:
            x = x[:,0]
        
        # Append these frames to the window.
        X.extend(x)
        
        # Perform the FFT.
        y = np.fft.rfft(X * hann)
        
        # Only data up to window_size/2 is useful; the rest is past the Nyquist cutoff.
        y = y[:window_size/2]
        
        # Normalize by obtaining magnitude, multiplying by 2 (since we discarded half the FFT) and dividing by the window size.
        y = np.absolute(y) * 2.0 / np.sum(hann)
        
        # Scale according to reference level, which depends on the sample width.
        y = y / np.power(2.0, (8*width - 1))
        
        # Convert to dB.
        y = (20.0 * np.log10(y)).clip(-120)
        
        # Add this DFT frame to the output, update the progress bar, and truncate the window.
        Y.append(y)
        acc += window_size
        bar.update(acc)
        X = X[window_size / overlap:]
        
    # Inform progress bar that the computation is complete.
    bar.finish()

    # Time domain: We have len(Y) windows, so convert to seconds by multiplying
    # by window size, dividing by sample rate, and dividing by the overlap rate.
    t = np.arange(0, len(Y), dtype=np.float) * window_size / sample_freq / overlap
    
    # Frequency domain: There are window_size/2 frequencies represented, and we scale
    # by dividing by window size and multiplying by sample frequency.
    f = np.arange(0, window_size / 2, dtype=np.float) * sample_freq / window_size
    
    # By default, numpy will arrange the matrix by rows, so we need to transpose it.
    Y = np.array(Y).transpose()
    
    # Plot the spectrogram.
    ax = plt.subplot(111)
    plt.pcolormesh(t, f, Y, vmin=-120, vmax=0)
    
    # Use log scale above 100 Hz, linear below.
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
