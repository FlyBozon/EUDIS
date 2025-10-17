import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
import librosa.display

def plot_fft(data, samplerate, filename, output_dir, max_freq=1000):
    n = len(data)
    fft_vals = np.fft.fft(data)
    fft_freqs = np.fft.fftfreq(n, 1 / samplerate)
    
    positive_freqs = fft_freqs[:n // 2]
    magnitude = np.abs(fft_vals[:n // 2])
    
    # ograniczamy do 1000 Hz
    mask = positive_freqs <= max_freq
    plt.figure(figsize=(10, 5))
    plt.plot(positive_freqs[mask], magnitude[mask])
    plt.title(f"FFT (0–{max_freq} Hz) — {filename}")
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.tight_layout()
    
    fft_path = os.path.join(output_dir, f"{filename}_fft.png")
    plt.savefig(fft_path, dpi=300)
    plt.close()
    print(f"Zapisano FFT: {fft_path}")


def plot_spectrogram(data, samplerate, filename, output_dir):
    plt.figure(figsize=(10, 5))
    plt.specgram(data, Fs=samplerate, NFFT=1024, noverlap=512, cmap='magma')
    plt.title(f"Spektrogram — {filename}")
    plt.xlabel("Czas [s]")
    plt.ylabel("Częstotliwość [Hz]")
    plt.colorbar(label='Intensywność [dB]')
    plt.tight_layout()
    
    spec_path = os.path.join(output_dir, f"{filename}_spectrogram.png")
    plt.savefig(spec_path, dpi=300)
    plt.close()
    print(f"Zapisano spektrogram: {spec_path}")


def plot_mfcc(y, sr, filename, output_dir):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='viridis')
    plt.colorbar(label="Wartość współczynnika MFCC")
    plt.title(f"MFCC — {filename}")
    plt.tight_layout()
    
    mfcc_path = os.path.join(output_dir, f"{filename}_mfcc.png")
    plt.savefig(mfcc_path, dpi=300)
    plt.close()
    print(f"Zapisano MFCC: {mfcc_path}")


def process_wav(filepath, output_dir):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    
    try:
        samplerate, data = wavfile.read(filepath)
        if len(data.shape) > 1:  # stereo -> mono
            data = data[:, 0]
        
        # FFT i spektrogram z surowych danych
        plot_fft(data, samplerate, filename, output_dir)
        plot_spectrogram(data, samplerate, filename, output_dir)
        
        # MFCC z librosa
        y, sr = librosa.load(filepath, sr=None, mono=True)
        plot_mfcc(y, sr, filename, output_dir)
        
    except Exception as e:
        print(f"Błąd przy {filepath}: {e}")


def main():
    if len(sys.argv) < 2:
        print("Użycie: python audio_analysis.py <ścieżka_do_pliku_lub_folderu>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.isdir(input_path):
        wav_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(".wav")]
        if not wav_files:
            print("Brak plików WAV w folderze.")
            sys.exit(0)
        for wav_file in wav_files:
            process_wav(wav_file, output_dir)
    elif os.path.isfile(input_path) and input_path.lower().endswith(".wav"):
        process_wav(input_path, output_dir)
    else:
        print("Podaj poprawny plik .wav lub folder zawierający pliki .wav.")

if __name__ == "__main__":
    main()
