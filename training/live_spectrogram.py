import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

samplerate = 44100
blocksize = 1024
seconds_in_view = 2.0
nfft = 1024

samples_in_buffer = int(seconds_in_view * samplerate)
audio_buffer = np.zeros(samples_in_buffer)

fig, ax = plt.subplots(figsize=(10, 5))
specgram_data = None

def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(status)
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]

def update_plot(frame):
    global specgram_data
    ax.clear()
    Pxx, freqs, bins, im = ax.specgram(
        audio_buffer,
        NFFT=nfft,
        Fs=samplerate,
        noverlap=nfft // 2,
        cmap='magma'
    )
    ax.set_ylim(0, 1000)
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Częstotliwość [Hz]")
    ax.set_title("Spektrogram na żywo")
    return im,

stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate, blocksize=blocksize)

ani = FuncAnimation(fig, update_plot, interval=100, blit=False)

print("🎙️ Uruchamiam wizualizację spektrogramu (CTRL+C aby przerwać)...")
with stream:
    plt.show()