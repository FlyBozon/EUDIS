import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import pickle
import librosa
from scipy.io import wavfile
from collections import deque
import time


def extract_fft_features(data, samplerate, max_freq=1000, n_bins=100):
    """Extract FFT features up to max_freq"""
    n = len(data)
    fft_vals = np.fft.fft(data)
    fft_freqs = np.fft.fftfreq(n, 1 / samplerate)

    positive_freqs = fft_freqs[:n // 2]
    magnitude = np.abs(fft_vals[:n // 2])

    mask = positive_freqs <= max_freq
    freq_filtered = positive_freqs[mask]
    mag_filtered = magnitude[mask]

    bins = np.linspace(0, max_freq, n_bins + 1)
    fft_features = []
    for i in range(n_bins):
        mask_bin = (freq_filtered >= bins[i]) & (freq_filtered < bins[i + 1])
        if np.any(mask_bin):
            fft_features.append(np.mean(mag_filtered[mask_bin]))
        else:
            fft_features.append(0)

    return np.array(fft_features)

def extract_spectrogram_features(data, samplerate, n_fft=1024, hop_length=512):
    """Extract spectrogram features"""
    D = librosa.stft(data.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)

    spec_mean = np.mean(magnitude_db, axis=1)
    spec_std = np.std(magnitude_db, axis=1)

    return np.concatenate([spec_mean, spec_std])

def extract_mfcc_features(y, sr, n_mfcc=13):
    """Extract MFCC features"""
    mfccs = librosa.feature.mfcc(y=y.astype(np.float32), sr=sr, n_mfcc=n_mfcc)

    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    mfcc_max = np.max(mfccs, axis=1)
    mfcc_min = np.min(mfccs, axis=1)

    return np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])

def extract_all_features(data, samplerate):
    """Extract all features from audio data"""
    try:
        fft_feat = extract_fft_features(data, samplerate)
        spec_feat = extract_spectrogram_features(data, samplerate)
        mfcc_feat = extract_mfcc_features(data, samplerate)

        features = np.concatenate([fft_feat, spec_feat, mfcc_feat])
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


class DroneDetectionNN(torch.nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes=[512, 256, 128]):
        super(DroneDetectionNN, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(prev_size, hidden_size))
            layers.append(torch.nn.BatchNorm1d(hidden_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.3))
            prev_size = hidden_size

        layers.append(torch.nn.Linear(prev_size, num_classes))

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class RealtimeDroneDetector:
    def __init__(self, model_path='best_drone_model.pth',
                 scaler_path='scaler.pkl',
                 label_mapping_path='label_mapping.pkl',
                 confidence_threshold=0.7,
                 detection_buffer_size=5):

        print("Loading model components...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        with open(label_mapping_path, 'rb') as f:
            self.label_mapping = pickle.load(f)

        self.inv_label_mapping = {v: k for k, v in self.label_mapping.items()}

        self.drone_classes = []
        for class_name, class_idx in self.label_mapping.items():
            if class_name.lower() not in ['no_drone', 'unknown', 'nodrone', 'no drone']:
                self.drone_classes.append(class_idx)

        print(f"Drone classes: {[self.inv_label_mapping[i] for i in self.drone_classes]}")

        num_classes = len(self.label_mapping)
        input_size = self.scaler.n_features_in_
        self.model = DroneDetectionNN(input_size, num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.confidence_threshold = confidence_threshold
        self.detection_buffer = deque(maxlen=detection_buffer_size)

        self.last_prediction = None
        self.last_confidence = 0.0
        self.last_probabilities = None
        self.detection_count = 0

        print(f"✓ Model loaded on {self.device}")
        print(f"✓ Confidence threshold: {confidence_threshold}")
        print(f"✓ Detection buffer size: {detection_buffer_size}")

    def predict(self, audio_data, samplerate):
        """Predict drone detection from audio data"""
        features = extract_all_features(audio_data, samplerate)
        if features is None:
            return None, 0.0, None, False

        features_normalized = self.scaler.transform(features.reshape(1, -1))

        features_tensor = torch.FloatTensor(features_normalized).to(self.device)

        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = self.inv_label_mapping[predicted.item()]
        confidence_value = confidence.item()
        all_probs = probabilities.cpu().numpy()[0]

        is_drone = False
        if predicted.item() in self.drone_classes and confidence_value >= self.confidence_threshold:
            self.detection_buffer.append(1)
            if sum(self.detection_buffer) >= len(self.detection_buffer) * 0.6:
                is_drone = True
        else:
            self.detection_buffer.append(0)

        self.last_prediction = predicted_class
        self.last_confidence = confidence_value
        self.last_probabilities = all_probs

        if is_drone:
            self.detection_count += 1

        return predicted_class, confidence_value, all_probs, is_drone


class DroneDetectionVisualizer:
    def __init__(self, samplerate=44100, blocksize=1024, seconds_in_view=2.0,
                 nfft=1024, analysis_interval=1.0):

        self.samplerate = samplerate
        self.blocksize = blocksize
        self.seconds_in_view = seconds_in_view
        self.nfft = nfft
        self.analysis_interval = analysis_interval

        self.samples_in_buffer = int(seconds_in_view * samplerate)
        self.audio_buffer = np.zeros(self.samples_in_buffer)

        self.detector = RealtimeDroneDetector()

        self.is_drone_detected = False
        self.last_analysis_time = time.time()

        self.fig, (self.ax_spec, self.ax_info) = plt.subplots(2, 1,
                                                               figsize=(12, 8),
                                                               gridspec_kw={'height_ratios': [3, 1]})
        self.fig.patch.set_facecolor('black')

        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(0.5, 0.5, '', ha='center', va='center',
                                           fontsize=14, color='white',
                                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=samplerate,
            blocksize=blocksize
        )

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        self.audio_buffer = np.roll(self.audio_buffer, -frames)
        self.audio_buffer[-frames:] = indata[:, 0]

    def update_plot(self, frame):
        current_time = time.time()

        if current_time - self.last_analysis_time >= self.analysis_interval:
            self.analyze_audio()
            self.last_analysis_time = current_time

        self.ax_spec.clear()

        Pxx, freqs, bins, im = self.ax_spec.specgram(
            self.audio_buffer,
            NFFT=self.nfft,
            Fs=self.samplerate,
            noverlap=self.nfft // 2,
            cmap='magma'
        )

        self.ax_spec.set_ylim(0, 1000)
        self.ax_spec.set_xlabel("Czas [s]", color='white')
        self.ax_spec.set_ylabel("Częstotliwość [Hz]", color='white')

        if self.is_drone_detected:
            self.ax_spec.set_title("⚠️ DRON WYKRYTY! ⚠️",
                                   color='red', fontsize=16, fontweight='bold')
            self.ax_spec.set_facecolor('#330000')
        else:
            self.ax_spec.set_title("Spektrogram na żywo - Detekcja dronów",
                                   color='white', fontsize=14)
            self.ax_spec.set_facecolor('black')

        self.ax_spec.tick_params(colors='white')

        self.update_info_panel()

        return im,

    def analyze_audio(self):
        """Analyze current audio buffer for drone detection"""
        prediction, confidence, probs, is_drone = self.detector.predict(
            self.audio_buffer, self.samplerate
        )

        self.is_drone_detected = is_drone

        if is_drone:
            print(f"🚨 DRONE DETECTED: {prediction} (confidence: {confidence*100:.1f}%)")

    def update_info_panel(self):
        """Update the information panel"""
        if self.detector.last_prediction is None:
            info_str = "Inicjalizacja..."
            color = 'white'
        else:
            pred = self.detector.last_prediction
            conf = self.detector.last_confidence * 100

            if self.is_drone_detected:
                info_str = f"🚨 WYKRYTO: {pred.upper()}\n"
                info_str += f"Pewność: {conf:.1f}%\n"
                info_str += f"Łączna liczba detekcji: {self.detector.detection_count}"
                color = 'red'
            else:
                info_str = f"Status: Monitorowanie\n"
                info_str += f"Ostatnia predykcja: {pred}\n"
                info_str += f"Pewność: {conf:.1f}%"
                color = 'lime'

            if self.detector.last_probabilities is not None:
                info_str += "\n\nRozkład prawdopodobieństw:"
                for i, prob in enumerate(self.detector.last_probabilities):
                    class_name = self.detector.inv_label_mapping[i]
                    info_str += f"\n  {class_name}: {prob*100:.1f}%"

        self.info_text.set_text(info_str)
        self.info_text.set_color(color)

    def run(self):
        """Start the real-time detection"""
        print("\n" + "="*60)
        print("🎙️  SYSTEM DETEKCJI DRONÓW URUCHOMIONY")
        print("="*60)
        print(f"Sample rate: {self.samplerate} Hz")
        print(f"Analiza co {self.analysis_interval}s")
        print(f"Naciśnij CTRL+C aby zakończyć")
        print("="*60 + "\n")

        ani = FuncAnimation(self.fig, self.update_plot, interval=100, blit=False)

        with self.stream:
            plt.show()


def main():
    SAMPLERATE = 44100
    BLOCKSIZE = 1024
    SECONDS_IN_VIEW = 2.0
    NFFT = 1024
    ANALYSIS_INTERVAL = 1.0

    try:
        visualizer = DroneDetectionVisualizer(
            samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE,
            seconds_in_view=SECONDS_IN_VIEW,
            nfft=NFFT,
            analysis_interval=ANALYSIS_INTERVAL
        )
        visualizer.run()
    except KeyboardInterrupt:
        print("\n\n✓ Zatrzymano detekcję dronów")
    except FileNotFoundError as e:
        print(f"\n❌ Błąd: Nie znaleziono plików modelu!")
        print("Upewnij się, że w bieżącym katalogu znajdują się:")
        print("  - best_drone_model.pth")
        print("  - scaler.pkl")
        print("  - label_mapping.pkl")
        print(f"\nSzczegóły: {e}")
    except Exception as e:
        print(f"\n❌ Wystąpił błąd: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()