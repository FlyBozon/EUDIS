import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import pickle
import librosa
from scipy import signal
from collections import deque
import time
import cv2


class AdaptiveNoiseReducer:
    """
    Learns background noise profile and subtracts it from signal.
    This helps isolate drone sounds from environment noise.
    """
    def __init__(self, samplerate, learning_time=5.0, update_rate=0.1):
        self.samplerate = samplerate
        self.learning_time = learning_time
        self.update_rate = update_rate

        self.noise_profile = None
        self.noise_samples = []
        self.is_learning = True
        self.learning_start_time = time.time()

        print(f"🎧 Adaptive Noise Reducer initialized")
        print(f"   Learning background noise for {learning_time}s...")

    def add_noise_sample(self, audio_data):
        """Add sample to noise profile during learning phase"""
        if self.is_learning:
            fft_data = np.fft.rfft(audio_data)
            magnitude = np.abs(fft_data)
            self.noise_samples.append(magnitude)

            if time.time() - self.learning_start_time > self.learning_time:
                self._finalize_noise_profile()

    def _finalize_noise_profile(self):
        """Calculate average noise profile from collected samples"""
        if len(self.noise_samples) > 0:
            self.noise_profile = np.median(self.noise_samples, axis=0)
            self.is_learning = False
            print(f"✓ Noise profile learned from {len(self.noise_samples)} samples")
            print(f"✓ Now actively reducing background noise")
        else:
            print("⚠ No noise samples collected, disabling noise reduction")

    def update_noise_profile(self, audio_data, is_quiet=True):
        """
        Continuously update noise profile with quiet samples.
        Only update when no drone is detected.
        """
        if not self.is_learning and is_quiet and self.noise_profile is not None:
            fft_data = np.fft.rfft(audio_data)
            magnitude = np.abs(fft_data)

            self.noise_profile = (1 - self.update_rate) * self.noise_profile + \
                                 self.update_rate * magnitude

    def reduce_noise(self, audio_data):
        """Apply spectral subtraction to remove noise"""
        if self.noise_profile is None:
            return audio_data

        fft_data = np.fft.rfft(audio_data)
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)

        if len(magnitude) != len(self.noise_profile):
            self.noise_profile = np.interp(
                np.linspace(0, 1, len(magnitude)),
                np.linspace(0, 1, len(self.noise_profile)),
                self.noise_profile
            )

        alpha = 2.0
        beta = 0.1

        clean_magnitude = magnitude - alpha * self.noise_profile
        clean_magnitude = np.maximum(clean_magnitude, beta * magnitude)

        clean_fft = clean_magnitude * np.exp(1j * phase)
        clean_audio = np.fft.irfft(clean_fft, n=len(audio_data))

        return clean_audio


class SimpleDroneCNN(torch.nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super(SimpleDroneCNN, self).__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(inplace=True), torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(inplace=True), torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                      stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                      stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class DroneResNet(torch.nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super(DroneResNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def create_spectrogram_image(audio_data, samplerate, target_size=(224, 224)):
    """Convert audio to spectrogram image for CNN"""
    try:
        audio_data = np.array(audio_data, dtype=np.float64)
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)

        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < 1e-6:
            return None

        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
        audio_data = np.clip(audio_data, -1.0, 1.0)

        nyquist = samplerate / 2
        low = max(50 / nyquist, 0.01)
        high = min(1000 / nyquist, 0.99)

        try:
            b, a = signal.butter(5, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, audio_data)
            filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            filtered = audio_data

        mel_spec = librosa.feature.melspectrogram(
            y=filtered.astype(np.float32),
            sr=samplerate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            fmin=50,
            fmax=1000
        )

        mel_spec = np.maximum(mel_spec, 1e-10)

        mel_spec_db = 10 * np.log10(mel_spec + 1e-10)
        mel_spec_db = np.nan_to_num(mel_spec_db, nan=-80.0, posinf=0.0, neginf=-80.0)

        spec_min = np.percentile(mel_spec_db, 5)
        spec_max = np.percentile(mel_spec_db, 95)

        if spec_max - spec_min < 1.0:
            return None

        mel_norm = (mel_spec_db - spec_min) / (spec_max - spec_min)
        mel_norm = np.clip(mel_norm, 0.0, 1.0)

        spectrogram_image = cv2.resize(mel_norm, target_size, interpolation=cv2.INTER_LINEAR)

        return spectrogram_image

    except Exception as e:
        return None


class VisualDroneDetector:
    def __init__(self,
                 model_path='best_drone_visual_model3/best_drone_visual_model.pth',
                 config_path='best_drone_visual_model3/visual_model_config.pkl',
                 label_mapping_path='best_drone_visual_model3/label_mapping.pkl',
                 confidence_threshold=0.75,
                 samplerate=44100):

        print("Loading visual drone detection model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(config_path, 'rb') as f:
            config = pickle.load(f)

        self.model_type = config['model_type']
        self.use_multi_channel = config['use_multi_channel']
        self.target_size = config['target_size']
        self.input_channels = config['input_channels']
        num_classes = config['num_classes']

        with open(label_mapping_path, 'rb') as f:
            self.label_mapping = pickle.load(f)

        self.inv_label_mapping = {v: k for k, v in self.label_mapping.items()}

        self.drone_classes = []
        for class_name, class_idx in self.label_mapping.items():
            if class_name.lower() not in ['no_drone', 'unknown', 'nodrone', 'no drone']:
                self.drone_classes.append(class_idx)

        if self.model_type == 'resnet':
            self.model = DroneResNet(num_classes, self.input_channels).to(self.device)
        else:
            self.model = SimpleDroneCNN(num_classes, self.input_channels).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.noise_reducer = AdaptiveNoiseReducer(samplerate, learning_time=5.0)

        self.confidence_threshold = confidence_threshold
        self.detection_buffer = deque(maxlen=5)

        self.last_prediction = None
        self.last_confidence = 0.0
        self.last_probabilities = None
        self.detection_count = 0
        self.last_spectrogram = None
        self.last_spectrogram_raw = None

        print(f"OK Model: {self.model_type} | Device: {self.device}")
        print(f"OK Drone classes: {[self.inv_label_mapping[i] for i in self.drone_classes]}")

    def predict(self, audio_data, samplerate):
        """Predict using visual CNN analysis with noise reduction"""

        if self.noise_reducer.is_learning:
            self.noise_reducer.add_noise_sample(audio_data)
            return "learning", 0.0, None, False, None, None

        clean_audio = self.noise_reducer.reduce_noise(audio_data)

        spec_image_raw = create_spectrogram_image(audio_data, samplerate, self.target_size)
        spec_image = create_spectrogram_image(clean_audio, samplerate, self.target_size)

        if spec_image is None:
            self.noise_reducer.update_noise_profile(audio_data, is_quiet=True)
            return None, 0.0, None, False, None, None

        self.last_spectrogram = spec_image
        self.last_spectrogram_raw = spec_image_raw

        spec_tensor = torch.FloatTensor(spec_image).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(spec_tensor)
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
            self.noise_reducer.update_noise_profile(audio_data, is_quiet=True)

        self.last_prediction = predicted_class
        self.last_confidence = confidence_value
        self.last_probabilities = all_probs

        if is_drone:
            self.detection_count += 1

        return predicted_class, confidence_value, all_probs, is_drone, spec_image, spec_image_raw


class DroneDetectionVisualizer:
    def __init__(self, samplerate=44100, blocksize=1024, seconds_in_view=2.0,
                 nfft=1024, analysis_interval=0.5):

        self.samplerate = samplerate
        self.blocksize = blocksize
        self.seconds_in_view = seconds_in_view
        self.nfft = nfft
        self.analysis_interval = analysis_interval

        self.samples_in_buffer = int(seconds_in_view * samplerate)
        self.audio_buffer = np.zeros(self.samples_in_buffer, dtype=np.float64)

        self.detector = VisualDroneDetector(samplerate=samplerate)

        self.is_drone_detected = False
        self.last_analysis_time = time.time()

        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(3, 3, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)

        self.ax_spec_live = self.fig.add_subplot(gs[0, :])
        self.ax_spec_raw = self.fig.add_subplot(gs[1, 0])
        self.ax_spec_clean = self.fig.add_subplot(gs[1, 1])
        self.ax_probs = self.fig.add_subplot(gs[1, 2])
        self.ax_info = self.fig.add_subplot(gs[2, :])

        self.fig.patch.set_facecolor('black')

        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(0.5, 0.5, '', ha='center', va='center',
                                           fontsize=10, color='white', family='monospace',
                                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=samplerate,
            blocksize=blocksize
        )

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            pass

        try:
            data = indata[:, 0].copy()
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            self.audio_buffer = np.roll(self.audio_buffer, -frames)
            self.audio_buffer[-frames:] = data
        except Exception:
            pass

    def update_plot(self, frame):
        current_time = time.time()

        if current_time - self.last_analysis_time >= self.analysis_interval:
            self.analyze_audio()
            self.last_analysis_time = current_time

        for ax in [self.ax_spec_live, self.ax_spec_raw, self.ax_spec_clean, self.ax_probs]:
            ax.clear()

        try:
            Pxx, freqs, bins, im = self.ax_spec_live.specgram(
                self.audio_buffer, NFFT=self.nfft, Fs=self.samplerate,
                noverlap=self.nfft // 2, cmap='magma'
            )
            self.ax_spec_live.set_ylim(0, 1000)
        except Exception:
            pass

        self.ax_spec_live.set_xlabel("Czas [s]", color='white', fontsize=9)
        self.ax_spec_live.set_ylabel("Czestotliwosc [Hz]", color='white', fontsize=9)

        if self.is_drone_detected:
            self.ax_spec_live.set_title("DRON WYKRYTY!", color='red', fontsize=16, fontweight='bold')
            self.ax_spec_live.set_facecolor('#330000')
        else:
            self.ax_spec_live.set_title("Spektrogram Live + Redukcja Szumu", color='lime', fontsize=14)
            self.ax_spec_live.set_facecolor('black')

        self.ax_spec_live.tick_params(colors='white', labelsize=8)

        if self.detector.last_spectrogram_raw is not None:
            self.ax_spec_raw.imshow(self.detector.last_spectrogram_raw,
                                   aspect='auto', origin='lower', cmap='magma')
            self.ax_spec_raw.set_title("Przed (z szumem)", color='yellow', fontsize=11)
        else:
            self.ax_spec_raw.text(0.5, 0.5, 'Oczekiwanie...', ha='center', va='center',
                                 color='white', fontsize=12, transform=self.ax_spec_raw.transAxes)

        self.ax_spec_raw.set_facecolor('black')
        self.ax_spec_raw.tick_params(colors='white', labelsize=7)
        self.ax_spec_raw.set_xlabel("Czas", color='white', fontsize=8)
        self.ax_spec_raw.set_ylabel("Freq", color='white', fontsize=8)

        if self.detector.last_spectrogram is not None:
            self.ax_spec_clean.imshow(self.detector.last_spectrogram,
                                     aspect='auto', origin='lower', cmap='magma')
            self.ax_spec_clean.set_title("Po (oczyszczony) -> CNN", color='lime', fontsize=11)
        else:
            self.ax_spec_clean.text(0.5, 0.5, 'Oczekiwanie...', ha='center', va='center',
                                   color='white', fontsize=12, transform=self.ax_spec_clean.transAxes)

        self.ax_spec_clean.set_facecolor('black')
        self.ax_spec_clean.tick_params(colors='white', labelsize=7)
        self.ax_spec_clean.set_xlabel("Czas", color='white', fontsize=8)
        self.ax_spec_clean.set_ylabel("Freq", color='white', fontsize=8)

        if self.detector.last_probabilities is not None:
            classes = [self.detector.inv_label_mapping[i]
                      for i in range(len(self.detector.last_probabilities))]
            probs = self.detector.last_probabilities * 100

            colors = ['red' if i in self.detector.drone_classes else 'green'
                     for i in range(len(probs))]

            bars = self.ax_probs.barh(classes, probs, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
            self.ax_probs.set_xlim(0, 100)
            self.ax_probs.set_xlabel('%', color='white', fontsize=9)
            self.ax_probs.set_title('CNN Output', color='white', fontsize=11)
            self.ax_probs.tick_params(colors='white', labelsize=8)

            for i, (bar, prob) in enumerate(zip(bars, probs)):
                self.ax_probs.text(prob + 2, i, f'{prob:.1f}%',
                                  va='center', color='white', fontsize=9, fontweight='bold')

        self.ax_probs.set_facecolor('black')
        for spine in self.ax_probs.spines.values():
            spine.set_edgecolor('white')

        self.update_info_panel()

    def analyze_audio(self):
        """Analyze audio"""
        try:
            result = self.detector.predict(self.audio_buffer, self.samplerate)

            if result[0] == "learning":
                return

            if result[0] is not None:
                pred, conf, probs, is_drone, spec, spec_raw = result
                self.is_drone_detected = is_drone

                if is_drone:
                    print(f"DRON: {pred} ({conf*100:.1f}%)")
        except Exception:
            pass

    def update_info_panel(self):
        """Update info"""
        if self.detector.noise_reducer.is_learning:
            elapsed = time.time() - self.detector.noise_reducer.learning_start_time
            remaining = max(0, self.detector.noise_reducer.learning_time - elapsed)
            info_str = f"UCZENIE PROFILU SZUMU...\n"
            info_str += f"Pozostalo: {remaining:.1f}s\n"
            info_str += f"Probki: {len(self.detector.noise_reducer.noise_samples)}"
            color = 'yellow'
        elif self.detector.last_prediction is None:
            info_str = "Inicjalizacja..."
            color = 'yellow'
        else:
            pred = self.detector.last_prediction
            conf = self.detector.last_confidence * 100

            if self.is_drone_detected:
                info_str = f"{'='*80}\n"
                info_str += f"  DRON WYKRYTY: {pred.upper()}\n"
                info_str += f"{'='*80}\n"
                info_str += f"Model: {self.detector.model_type.upper()} | "
                info_str += f"Pewnosc: {conf:.1f}% | "
                info_str += f"Detekcje: {self.detector.detection_count}\n"
                color = 'red'
            else:
                info_str = f"OK MONITORING (Redukcja szumu aktywna)\n"
                info_str += f"{'─'*80}\n"
                info_str += f"Model: {self.detector.model_type} | "
                info_str += f"Predykcja: {pred} | Pewnosc: {conf:.1f}%"
                color = 'lime'

        self.info_text.set_text(info_str)
        self.info_text.set_color(color)

    def run(self):
        """Start detection"""
        print("\n" + "="*80)
        print("SYSTEM DETEKCJI DRONOW Z ADAPTACYJNA REDUKCJA SZUMU")
        print("="*80)
        print(f"OK Model: {self.detector.model_type.upper()}")
        print(f"OK Redukcja szumu: Adaptacyjna (uczenie + ciagla aktualizacja)")
        print(f"OK Analiza co: {self.analysis_interval}s")
        print("="*80)
        print("\nSystem automatycznie:")
        print("  1. Uczy sie szumu tla przez pierwsze 5s")
        print("  2. Odejmuje szum srodowiska od sygnalu")
        print("  3. Aktualizuje profil szumu gdy brak dronow")
        print("="*80)
        print("\nNacisnij CTRL+C aby zakonczyc\n")

        ani = FuncAnimation(self.fig, self.update_plot, interval=100,
                          blit=False, cache_frame_data=False)

        with self.stream:
            plt.show()


def main():
    SAMPLERATE = 44100
    BLOCKSIZE = 1024
    SECONDS_IN_VIEW = 2.0
    NFFT = 1024
    ANALYSIS_INTERVAL = 0.5

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
        print("\n\nSystem zatrzymany")
    except FileNotFoundError as e:
        print(f"\nBlad: Brak plikow modelu!")
        print("Wymagane pliki w folderze 'best_drone_visual_model1/':")
        print("  - best_drone_visual_model.pth")
        print("  - visual_model_config.pkl")
        print("  - label_mapping.pkl")
        print(f"\n{e}")
    except Exception as e:
        print(f"\nBlad: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()