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
    dtype = np.int8 if width == 1 else np.int16
    
    # Y will hold the DFT of each window. We use acc and bar for displaying progress.
    Y = []
    acc = 0
    bar = ProgressBar(max_value=frame)
    
    # Process each window of audio.
    while True:
        
        try:
            x = w.readframes(window_size)
        except:
            break
            
        # Load raw audio data into a numpy array.
        x = np.fromstring(x, dtype=dtype)
        
        # If the window read was short, end.
        if len(x) != window_size * chans:
            break
        
        # Reshape our array into an N*c matrix in order to separate the channels.
        x = np.reshape(x, (window_size, chans))

        # Average the channels if the audio is stereo.
        if chans > 1:
            x = (x[:,0] + x[:,1]) / 2
        
        # Perform the FFT.
        y = np.fft.fft(x)
        
        # Only data up to window_size/2 is useful; the rest is past the Nyquist cutoff.
        y = y[:window_size/2]
        
        # Normalize the data by converting to dB.
        y = (np.log10(np.power(np.absolute(y), 2).clip(1)) * 10)
        
        # Add this DFT frame to the output and update the progress bar.
        Y.append(y)
        acc += window_size
        bar.update(acc)
        
    # Inform progress bar that the computation is complete.
    bar.finish()

    # Time domain: We have len(Y) windows, so convert to seconds by multiplying
    # by window size and dividing by sample rate.
    t = np.arange(0, len(Y), dtype=np.float) * window_size / sample_freq
    
    # Frequency domain: There are window_size/2 frequencies represented, and we scale
    # by dividing by window size and multiplying by sample frequency.
    f = np.arange(0, window_size / 2, dtype=np.float) * sample_freq / window_size
    
    # By default, numpy will arrange the matrix by rows, so we need to transpose it.
    Y = np.array(Y).transpose()
    
    # Scale dB by maximum value.
    m = np.amax(Y)
    Y = Y - m
    
    # Plot the spectrogram.
    ax = plt.subplot(111)
    plt.pcolormesh(t, f, Y)
    
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
