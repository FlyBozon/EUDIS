import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.signal as sps
import time, os

SR = 44100
BLOCK = 1024
LOW, HIGH = 50, 1000
WINDOW_SEC = 1.0
OUTPUT_DIR = "detections"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def bandpass_filter(x, sr, low, high, order=4):
    nyq = sr / 2
    b, a = sps.butter(order, [low / nyq, high / nyq], btype='band')
    return sps.lfilter(b, a, x)

def compute_features(sig, sr, bg_spectrum=None):
    N = len(sig)
    fft_vals = np.fft.rfft(sig * np.hanning(N))
    freqs = np.fft.rfftfreq(N, 1 / sr)
    mag = np.abs(fft_vals)

    if bg_spectrum is not None:
        mag = np.maximum(mag - bg_spectrum, 0)

    band = (freqs >= LOW) & (freqs <= HIGH)
    band_mag = mag[band]

    rms = np.sqrt(np.mean(sig ** 2))
    peaks, _ = sps.find_peaks(band_mag, height=np.max(band_mag) * 0.3, distance=8)
    harmonicity = len(peaks)
    eps = 1e-10
    flatness = np.exp(np.mean(np.log(band_mag + eps))) / (np.mean(band_mag) + eps)
    score = rms * (1 - flatness) + harmonicity * 0.05
    return score, rms, harmonicity, flatness, freqs, mag

def compute_drone_features(sig, sr, bg_spectrum=None):
    N = len(sig)
    fft_vals = np.fft.rfft(sig * np.hanning(N))
    freqs = np.fft.rfftfreq(N, 1 / sr)
    mag = np.abs(fft_vals)

    if bg_spectrum is not None:
        mag = np.maximum(mag - bg_spectrum, 0)

    band = (freqs >= LOW) & (freqs <= HIGH)
    band_mag = mag[band]
    band_freqs = freqs[band]

    peaks, _ = sps.find_peaks(band_mag, height=np.max(band_mag)*0.3, distance=10)
    harmonicity = len(peaks)
    if len(peaks) > 1:
        distances = np.diff(peaks)
        harmonicity_score = 1 / (np.std(distances)+1e-6)
    else:
        harmonicity_score = 0

    eps = 1e-10
    flatness = np.exp(np.mean(np.log(band_mag + eps))) / (np.mean(band_mag) + eps)
    tonal_score = 1 - flatness

    ceps = np.fft.irfft(np.log(band_mag + eps))
    peak_ceps = np.max(np.abs(ceps[1:int(len(ceps)/2)]))
    periodicity_score = peak_ceps

    score = harmonicity_score*0.5 + tonal_score*0.3 + periodicity_score*0.2
    return score, harmonicity_score, tonal_score, periodicity_score


def is_drone(score, threshold=0.1):
    return score > threshold

def live_drone_detector():
    buffer = np.zeros(int(SR * 5))
    bg_spectrum = None
    start_time = time.time()
    detections = []

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_ylim(0, HIGH)
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Częstotliwość [Hz]")
    ax.set_title("Spektrogram na żywo + detekcja drona")

    spec_img = None
    detected_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                            color='red', fontsize=14, fontweight='bold')

    def audio_callback(indata, frames, time_info, status):
        nonlocal buffer
        buffer = np.roll(buffer, -frames)
        buffer[-frames:] = indata[:, 0]

    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SR, blocksize=BLOCK)

    def update(frame):
        nonlocal bg_spectrum, spec_img

        elapsed = time.time() - start_time
        sig = buffer[-int(WINDOW_SEC * SR):]
        sig_f = bandpass_filter(sig, SR, LOW, HIGH)

        if 2 < elapsed < 7:
            fft_bg_new = np.abs(np.fft.rfft(sig_f * np.hanning(len(sig_f))))
            if bg_spectrum is None:
                bg_spectrum = fft_bg_new
            else:
                bg_spectrum = 0.9 * bg_spectrum + 0.1 * fft_bg_new

        score, harmonicity_score, tonal_score, periodicity_score = compute_drone_features(sig_f, SR, bg_spectrum)

        Pxx, freqs_spec, bins, im = ax.specgram(sig_f, NFFT=1024, Fs=SR, noverlap=512, cmap='magma')
        if spec_img is None:
            spec_img = im
        else:
            spec_img.set_array(10*np.log10(Pxx+1e-10))

        ax.set_ylim(0, HIGH)
        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Częstotliwość [Hz]")
        ax.set_title("Spektrogram na żywo + detekcja drona")

        if is_drone(score):
            msg = (f"🚁 DRON wykryty! score={score:.3f} | "
                f"H={harmonicity_score:.2f} T={tonal_score:.2f} P={periodicity_score:.2f}")
            print(msg)
            detected_text.set_text(msg)
            detected_text.set_color('red')

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(OUTPUT_DIR, f"detection_{timestamp}.png")
            plt.savefig(save_path, dpi=200)
            detections.append((timestamp, score))
        else:
            detected_text.set_text("Brak drona")
            detected_text.set_color('green')

        return spec_img,

    ani = FuncAnimation(fig, update, interval=500, blit=False)

    print("🎙️ Tryb detekcji na żywo (2–7 s = uczenie tła)... CTRL+C aby zakończyć.")
    with stream:
        plt.show()

    if detections:
        txt_path = os.path.join(OUTPUT_DIR, "detections.txt")
        with open(txt_path, "w") as f:
            for t, s in detections:
                f.write(f"{t}\t{s:.3f}\n")
        print(f"✅ Zapisano listę wykryć: {txt_path}")
    else:
        print("Brak wykryć drona.")

if __name__ == "__main__":
    live_drone_detector()