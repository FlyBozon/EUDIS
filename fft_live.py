import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons, RadioButtons
from scipy import signal
from scipy.signal import find_peaks
import json
import os
from datetime import datetime

# parametry
samplerate = 44100       # Hz
blocksize = 1024         # próbki na blok
seconds_in_view = 2.0    # ile sekund widzimy na spektrogramie
nfft = 4096              # wyższa rozdzielczość FFT dla lepszej precyzji

# bufor danych
samples_in_buffer = int(seconds_in_view * samplerate)
audio_buffer = np.zeros(samples_in_buffer)

# Klasa dla profilu drona
class DroneProfile:
    def __init__(self, name, target_frequency, harmonics=None, metadata=None):
        self.name = name
        self.target_frequency = target_frequency
        self.frequency_tolerance = 20.0
        self.harmonics = harmonics if harmonics else []  # Lista rzeczywistych częstotliwości harmonicznych
        self.harmonic_multiples = []  # [2, 3, 4, ...] - wielokrotności
        self.harmonic_tolerance = 30.0
        self.metadata = metadata if metadata else {}
        self.created_at = metadata.get('created_at', datetime.now().isoformat())
        self.confidence = metadata.get('confidence', 0.0)
        
    def to_dict(self):
        return {
            'name': self.name,
            'target_frequency': self.target_frequency,
            'frequency_tolerance': self.frequency_tolerance,
            'harmonics': self.harmonics,
            'harmonic_multiples': self.harmonic_multiples,
            'harmonic_tolerance': self.harmonic_tolerance,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'confidence': self.confidence
        }
    
    @staticmethod
    def from_dict(data):
        profile = DroneProfile(
            name=data['name'],
            target_frequency=data['target_frequency'],
            harmonics=data.get('harmonics', []),
            metadata=data.get('metadata', {})
        )
        profile.frequency_tolerance = data.get('frequency_tolerance', 20.0)
        profile.harmonic_multiples = data.get('harmonic_multiples', [])
        profile.harmonic_tolerance = data.get('harmonic_tolerance', 30.0)
        profile.created_at = data.get('created_at', datetime.now().isoformat())
        profile.confidence = data.get('confidence', 0.0)
        return profile

# Manager profili dronów
class DroneProfileManager:
    def __init__(self, filename='drone_profiles.json'):
        self.filename = filename
        self.profiles = []
        self.active_profile_index = None
        self.load_profiles()
        
    def add_profile(self, profile):
        # Sprawdź czy profil o takiej nazwie już istnieje
        existing = self.get_profile_by_name(profile.name)
        if existing:
            # Aktualizuj istniejący
            idx = self.profiles.index(existing)
            self.profiles[idx] = profile
            print(f"📝 Zaktualizowano profil: {profile.name}")
        else:
            self.profiles.append(profile)
            print(f"✅ Dodano nowy profil: {profile.name}")
        self.save_profiles()
        
    def remove_profile(self, name):
        profile = self.get_profile_by_name(name)
        if profile:
            self.profiles.remove(profile)
            self.save_profiles()
            print(f"🗑️ Usunięto profil: {name}")
            return True
        return False
    
    def get_profile_by_name(self, name):
        for profile in self.profiles:
            if profile.name == name:
                return profile
        return None
    
    def get_active_profile(self):
        if self.active_profile_index is not None and 0 <= self.active_profile_index < len(self.profiles):
            return self.profiles[self.active_profile_index]
        return None
    
    def set_active_profile(self, index):
        if 0 <= index < len(self.profiles):
            self.active_profile_index = index
            return self.profiles[index]
        return None
    
    def get_profile_names(self):
        return [p.name for p in self.profiles]
    
    def save_profiles(self):
        try:
            data = [p.to_dict() for p in self.profiles]
            with open(self.filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"💾 Zapisano {len(self.profiles)} profili do {self.filename}")
        except Exception as e:
            print(f"❌ Błąd zapisu profili: {e}")
    
    def load_profiles(self):
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                self.profiles = [DroneProfile.from_dict(p) for p in data]
                print(f"📂 Wczytano {len(self.profiles)} profili z {self.filename}")
                if self.profiles:
                    self.active_profile_index = 0
            else:
                print(f"ℹ️ Brak pliku profili, tworzę nowy")
                self.profiles = []
        except Exception as e:
            print(f"❌ Błąd wczytywania profili: {e}")
            self.profiles = []

# Inicjalizacja managera profili
profile_manager = DroneProfileManager()

# Parametry detekcji drona
class DroneDetectionParams:
    def __init__(self):
        # Parametry ogólne
        self.detect_harmonics = True
        self.num_harmonics = 5
        self.min_harmonics_detected = 2
        self.require_fundamental = True
        
        # Próg amplitudy
        self.peak_threshold_multiplier = 2.0
        self.min_peak_amplitude = 100.0
        
        # Stabilność czasowa
        self.stability_window = 5
        self.stability_threshold = 0.7
        
        # Wagi komponentów
        self.weight_fundamental = 0.4
        self.weight_harmonics = 0.4
        self.weight_amplitude = 0.2
        
        # Próg końcowej detekcji
        self.detection_threshold = 0.6
        
        # Historia
        self.history_length = 5
        
        # Tryb detekcji
        self.detection_mode = 'single'  # 'single' lub 'multi'

params = DroneDetectionParams()

class DroneSignatureAnalyzer:
    """Analizuje sygnał i wyciąga charakterystyczne częstotliwości drona"""
    
    def __init__(self, samplerate):
        self.samplerate = samplerate
        self.is_analyzing = False
    
    def analyze_drone_signature(self, data):
        """
        Analizuje bufor audio i wyciąga charakterystykę drona
        """
        if self.is_analyzing:
            return None
        
        self.is_analyzing = True
        
        try:
            if len(data) < self.samplerate:
                return None
            
            # FFT
            fft_data = np.fft.rfft(data)
            freqs = np.fft.rfftfreq(len(data), 1/self.samplerate)
            magnitude = np.abs(fft_data)
            
            # Ogranicz do sensownego zakresu (50-5000 Hz)
            mask = (freqs >= 50) & (freqs <= 5000)
            freqs = freqs[mask]
            magnitude = magnitude[mask]
            
            # Znajdź wszystkie znaczące piki
            avg_magnitude = np.mean(magnitude)
            std_magnitude = np.std(magnitude)
            threshold = avg_magnitude + 2 * std_magnitude
            
            peaks, properties = find_peaks(
                magnitude,
                height=threshold,
                distance=10,
                prominence=avg_magnitude * 0.5
            )
            
            if len(peaks) < 2:
                return None
            
            # Sortuj piki po amplitudzie
            peak_freqs = freqs[peaks]
            peak_amps = magnitude[peaks]
            sorted_indices = np.argsort(peak_amps)[::-1]
            
            # Weź top 10 najsilniejszych pików
            top_peaks = [(peak_freqs[i], peak_amps[i]) for i in sorted_indices[:10]]
            
            # Spróbuj znaleźć częstotliwość podstawową
            best_fundamental = None
            best_harmonics = []
            best_score = 0
            
            for candidate_freq, candidate_amp in top_peaks:
                if candidate_freq < 50 or candidate_freq > 2000:
                    continue
                
                # Sprawdź ile harmonicznych tego kandydata istnieje
                harmonics = []
                for n in range(2, 8):
                    expected = candidate_freq * n
                    if expected > freqs[-1]:
                        break
                    
                    tolerance = 50
                    close_peaks = [f for f, a in top_peaks 
                                 if abs(f - expected) < tolerance and a > avg_magnitude * 1.5]
                    
                    if close_peaks:
                        harmonics.append((n, close_peaks[0]))
                
                score = len(harmonics) * (candidate_amp / avg_magnitude)
                
                if score > best_score:
                    best_score = score
                    best_fundamental = candidate_freq
                    best_harmonics = harmonics
            
            if best_fundamental is None:
                best_fundamental = top_peaks[0][0]
                best_harmonics = []
            
            confidence = min(best_score / 10.0, 1.0)
            
            return {
                'fundamental': best_fundamental,
                'harmonics': [h[1] for h in best_harmonics],
                'harmonic_multiples': [h[0] for h in best_harmonics],
                'all_peaks': top_peaks,
                'confidence': confidence
            }
        finally:
            self.is_analyzing = False

analyzer = DroneSignatureAnalyzer(samplerate)

class MultiProfileDetector:
    """Detektor obsługujący wiele profili jednocześnie"""
    
    def __init__(self, samplerate):
        self.samplerate = samplerate
        self.detection_histories = {}  # {profile_name: [history]}
        
    def find_peak_near_frequency(self, freqs, magnitude, target_freq, tolerance):
        """Szuka piku w okolicy target_freq ± tolerance"""
        mask = (freqs >= target_freq - tolerance) & (freqs <= target_freq + tolerance)
        
        if not np.any(mask):
            return False, None, 0.0
        
        freqs_range = freqs[mask]
        magnitude_range = magnitude[mask]
        
        if len(magnitude_range) == 0:
            return False, None, 0.0
        
        max_idx = np.argmax(magnitude_range)
        max_freq = freqs_range[max_idx]
        max_amp = magnitude_range[max_idx]
        
        avg_magnitude = np.mean(magnitude)
        
        if max_amp > avg_magnitude * params.peak_threshold_multiplier and max_amp > params.min_peak_amplitude:
            return True, max_freq, max_amp
        
        return False, None, 0.0
    
    def detect_profile(self, data, profile):
        """Wykrywa konkretny profil drona"""
        try:
            # FFT
            fft_data = np.fft.rfft(data)
            freqs = np.fft.rfftfreq(len(data), 1/self.samplerate)
            magnitude = np.abs(fft_data)
            
            # Wykryj częstotliwość podstawową
            found, actual_freq, amplitude = self.find_peak_near_frequency(
                freqs, magnitude,
                profile.target_frequency,
                profile.frequency_tolerance
            )
            
            fundamental_score = 1.0 if found else 0.0
            
            # Wykryj harmoniczne
            harmonics_score = 0.0
            harmonics_info = []
            
            if params.detect_harmonics and found:
                # Sprawdź harmoniczne na podstawie rzeczywistej wykrytej częstotliwości
                for n in range(2, params.num_harmonics + 2):
                    expected_harmonic = actual_freq * n
                    
                    if expected_harmonic > freqs[-1]:
                        break
                    
                    h_found, h_actual_freq, h_amplitude = self.find_peak_near_frequency(
                        freqs, magnitude,
                        expected_harmonic,
                        profile.harmonic_tolerance
                    )
                    
                    if h_found:
                        harmonics_info.append({
                            'multiplier': n,
                            'expected': expected_harmonic,
                            'actual': h_actual_freq,
                            'amplitude': h_amplitude
                        })
                
                num_detected = len(harmonics_info)
                if num_detected >= params.min_harmonics_detected:
                    harmonics_score = min(num_detected / params.num_harmonics, 1.0)
            
            # Sprawdź amplitudy
            amplitude_score = 1.0
            if amplitude and amplitude > 0:
                for harmonic in harmonics_info:
                    ratio = harmonic['amplitude'] / amplitude
                    if ratio > 2.0:
                        amplitude_score *= 0.5
            else:
                amplitude_score = 0.0
            
            # Całkowity wynik
            if params.require_fundamental and not found:
                total_score = 0.0
            else:
                total_score = (
                    fundamental_score * params.weight_fundamental +
                    harmonics_score * params.weight_harmonics +
                    amplitude_score * params.weight_amplitude
                )
            
            # Historia dla tego profilu
            if profile.name not in self.detection_histories:
                self.detection_histories[profile.name] = []
            
            self.detection_histories[profile.name].append(total_score >= params.detection_threshold)
            if len(self.detection_histories[profile.name]) > params.history_length:
                self.detection_histories[profile.name].pop(0)
            
            # Stabilność
            history = self.detection_histories[profile.name]
            if len(history) >= params.stability_window:
                recent = history[-params.stability_window:]
                stability = sum(recent) / len(recent)
                stable_detection = stability >= params.stability_threshold
            else:
                stable_detection = total_score >= params.detection_threshold
            
            return {
                'profile_name': profile.name,
                'detected': stable_detection,
                'score': total_score,
                'fundamental_found': found,
                'fundamental_freq': actual_freq,
                'fundamental_amplitude': amplitude,
                'harmonics_found': len(harmonics_info),
                'harmonics_info': harmonics_info,
                'fundamental_score': fundamental_score,
                'harmonics_score': harmonics_score,
                'amplitude_score': amplitude_score,
            }
        except Exception as e:
            print(f"⚠️ Błąd w detekcji profilu {profile.name}: {e}")
            return {
                'profile_name': profile.name,
                'detected': False,
                'score': 0.0,
                'fundamental_found': False,
                'fundamental_freq': None,
                'fundamental_amplitude': 0,
                'harmonics_found': 0,
                'harmonics_info': [],
                'fundamental_score': 0,
                'harmonics_score': 0,
                'amplitude_score': 0,
            }
    
    def detect_all_profiles(self, data):
        """Wykrywa wszystkie profile"""
        results = []
        for profile in profile_manager.profiles:
            result = self.detect_profile(data, profile)
            results.append(result)
        return results
    
    def detect_active_profile(self, data):
        """Wykrywa tylko aktywny profil"""
        active_profile = profile_manager.get_active_profile()
        if active_profile:
            return self.detect_profile(data, active_profile)
        return None

detector = MultiProfileDetector(samplerate)

detection_info = {
    'mode': 'single',  # 'single' lub 'multi'
    'active_result': None,
    'all_results': []
}

def process_original(data):
    return data

def process_bandpass_active(data):
    """Filtr pasmowy wokół aktywnego profilu"""
    try:
        active_profile = profile_manager.get_active_profile()
        if active_profile:
            nyquist = samplerate / 2
            low = (active_profile.target_frequency - 100) / nyquist
            high = (active_profile.target_frequency + 100) / nyquist
            low = max(0.01, min(low, 0.99))
            high = max(0.01, min(high, 0.99))
            if low < high:
                b, a = signal.butter(4, [low, high], btype='band')
                return signal.filtfilt(b, a, data)
    except:
        pass
    return data

def process_show_all_profiles(data):
    """Pokazuje wszystkie profile razem"""
    try:
        fft_data = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), 1/samplerate)
        
        filtered_fft = np.zeros_like(fft_data, dtype=complex)
        
        for profile in profile_manager.profiles:
            # Dodaj częstotliwość podstawową
            mask = (freqs >= profile.target_frequency - profile.frequency_tolerance) & \
                   (freqs <= profile.target_frequency + profile.frequency_tolerance)
            filtered_fft[mask] = fft_data[mask]
            
            # Dodaj harmoniczne
            for n in range(2, params.num_harmonics + 2):
                harmonic_f = profile.target_frequency * n
                if harmonic_f < freqs[-1]:
                    mask = (freqs >= harmonic_f - profile.harmonic_tolerance) & \
                           (freqs <= harmonic_f + profile.harmonic_tolerance)
                    filtered_fft[mask] = fft_data[mask]
        
        return np.fft.irfft(filtered_fft, len(data))
    except:
        return data

def process_detection_viz(data):
    return data

processors = [
    (lambda data: process_original(data), "Oryginalny sygnał"),
    (lambda data: process_bandpass_active(data), "Bandpass (aktywny profil)"),
    (lambda data: process_show_all_profiles(data), "Wszystkie profile"),
    (lambda data: process_detection_viz(data), "Detekcja")
]

# Konfiguracja interfejsu
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(6, 4, hspace=0.5, wspace=0.3)

# Spektrogramy
ax_specs = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[0, 2]),
    fig.add_subplot(gs[0, 3])
]

# FFT Plot
ax_fft = fig.add_subplot(gs[1, :])

# Panel kontrolny
ax_controls = fig.add_subplot(gs[2:, :])
ax_controls.axis('off')

main_title = fig.suptitle("", fontsize=14, weight='bold')

# Slidery
sliders = {}

slider_positions = [
    ('num_harmonics', 'Liczba harmonicznych', 1, 10, params.num_harmonics, 1),
    ('min_harmonics_detected', 'Min. wykrytych harmonicznych', 0, 10, params.min_harmonics_detected, 1),
    
    ('peak_threshold_multiplier', 'Mnożnik progu piku', 0.5, 10.0, params.peak_threshold_multiplier, 0.1),
    ('min_peak_amplitude', 'Min. amplituda piku', 0, 10000, params.min_peak_amplitude, 10),
    
    ('weight_fundamental', 'Waga: podstawowa', 0.0, 1.0, params.weight_fundamental, 0.01),
    ('weight_harmonics', 'Waga: harmoniczne', 0.0, 1.0, params.weight_harmonics, 0.01),
    ('weight_amplitude', 'Waga: amplituda', 0.0, 1.0, params.weight_amplitude, 0.01),
    
    ('detection_threshold', 'Próg detekcji', 0.0, 1.0, params.detection_threshold, 0.01),
    ('stability_window', 'Okno stabilności', 1, 20, params.stability_window, 1),
    ('stability_threshold', 'Próg stabilności', 0.0, 1.0, params.stability_threshold, 0.01),
    ('history_length', 'Długość historii', 1, 20, params.history_length, 1),
]

for i, (name, label, vmin, vmax, vinit, valstep) in enumerate(slider_positions):
    col = i % 2
    row = i // 2
    
    ax_slider = plt.axes([0.15 + col * 0.5, 0.54 - row * 0.027, 0.3, 0.012])
    slider = Slider(ax_slider, label, vmin, vmax, valinit=vinit, valstep=valstep)
    sliders[name] = slider

# TextBox dla nazwy profilu
ax_profile_name = plt.axes([0.15, 0.73, 0.15, 0.02])
text_profile_name = TextBox(ax_profile_name, 'Nazwa profilu:', initial='Dron_1')

# RadioButtons dla wyboru profilu
ax_radio = plt.axes([0.7, 0.60, 0.25, 0.15])
ax_radio.set_title("Aktywny profil:", fontsize=10)

profile_names = profile_manager.get_profile_names()
if not profile_names:
    profile_names = ['(brak profili)']

radio = RadioButtons(ax_radio, profile_names)

def update_radio_buttons():
    """Aktualizuje listę profili w radio buttons"""
    global radio, ax_radio
    ax_radio.clear()
    ax_radio.set_title("Aktywny profil:", fontsize=10)
    
    profile_names = profile_manager.get_profile_names()
    if not profile_names:
        profile_names = ['(brak profili)']
    
    radio = RadioButtons(ax_radio, profile_names)
    radio.on_clicked(select_profile)
    
    # Zaznacz aktywny
    if profile_manager.active_profile_index is not None:
        active_profile = profile_manager.get_active_profile()
        if active_profile and active_profile.name in profile_names:
            radio.set_active(profile_names.index(active_profile.name))

def select_profile(label):
    """Wybiera profil z radio buttons"""
    if label == '(brak profili)':
        return
    
    for i, profile in enumerate(profile_manager.profiles):
        if profile.name == label:
            profile_manager.set_active_profile(i)
            print(f"✅ Aktywny profil: {label} ({profile.target_frequency:.1f} Hz)")
            break

radio.on_clicked(select_profile)

# Callback dla sliderów
def update_params(val):
    params.num_harmonics = int(sliders['num_harmonics'].val)
    params.min_harmonics_detected = int(sliders['min_harmonics_detected'].val)
    
    params.peak_threshold_multiplier = sliders['peak_threshold_multiplier'].val
    params.min_peak_amplitude = sliders['min_peak_amplitude'].val
    
    params.weight_fundamental = sliders['weight_fundamental'].val
    params.weight_harmonics = sliders['weight_harmonics'].val
    params.weight_amplitude = sliders['weight_amplitude'].val
    
    params.detection_threshold = sliders['detection_threshold'].val
    params.stability_window = int(sliders['stability_window'].val)
    params.stability_threshold = sliders['stability_threshold'].val
    params.history_length = int(sliders['history_length'].val)

for slider in sliders.values():
    slider.on_changed(update_params)

# Checkbox
ax_checkbox = plt.axes([0.4, 0.73, 0.25, 0.02])
check = CheckButtons(ax_checkbox, ['Wymagaj podstawowej', 'Wykrywaj harmoniczne', 'Tryb Multi-Profile'], 
                     [params.require_fundamental, params.detect_harmonics, params.detection_mode == 'multi'])

def update_checkboxes(label):
    if label == 'Wymagaj podstawowej':
        params.require_fundamental = not params.require_fundamental
    elif label == 'Wykrywaj harmoniczne':
        params.detect_harmonics = not params.detect_harmonics
    elif label == 'Tryb Multi-Profile':
        params.detection_mode = 'multi' if params.detection_mode == 'single' else 'single'
        print(f"🔄 Tryb detekcji: {params.detection_mode.upper()}")

check.on_clicked(update_checkboxes)

# PRZYCISK: JEST DRON (dodaje nowy profil)
ax_learn_drone = plt.axes([0.15, 0.15, 0.12, 0.035])
btn_learn_drone = Button(ax_learn_drone, '🎯 JEST DRON\n(nowy profil)', color='lightgreen', hovercolor='green')

def learn_drone_signature(event):
    """
    Analizuje aktualny bufor audio i tworzy nowy profil
    """
    if analyzer.is_analyzing:
        print("⏳ Analiza w toku, proszę czekać...")
        return
    
    profile_name = text_profile_name.text.strip()
    if not profile_name:
        print("❌ Podaj nazwę profilu!")
        return
    
    print("\n" + "="*50)
    print(f"🎓 UCZENIE SIĘ NOWEGO PROFILU: {profile_name}")
    print("="*50)
    
    buffer_copy = audio_buffer.copy()
    signature = analyzer.analyze_drone_signature(buffer_copy)
    
    if signature is None:
        print("❌ Nie udało się wykryć wyraźnego sygnału drona")
        print("💡 Spróbuj gdy dron jest bliżej lub dźwięk jest wyraźniejszy")
        return
    
    print(f"\n📊 WYKRYTA SYGNATURA:")
    print(f"   Częstotliwość podstawowa: {signature['fundamental']:.1f} Hz")
    print(f"   Liczba harmonicznych: {len(signature['harmonics'])}")
    print(f"   Harmoniczne wielokrotności: {signature['harmonic_multiples']}")
    print(f"   Pewność: {signature['confidence']:.1%}")
    
    if signature['harmonics']:
        print(f"   Harmoniczne [Hz]: {[f'{h:.1f}' for h in signature['harmonics']]}")
    
    # Twórz nowy profil
    new_profile = DroneProfile(
        name=profile_name,
        target_frequency=signature['fundamental'],
        harmonics=signature['harmonics'],
        metadata={
            'created_at': datetime.now().isoformat(),
            'confidence': signature['confidence'],
            'harmonic_multiples': signature['harmonic_multiples'],
            'all_peaks': [(f, float(a)) for f, a in signature['all_peaks'][:5]]
        }
    )
    new_profile.harmonic_multiples = signature['harmonic_multiples']
    
    # Ustaw tolerancje
    if signature['confidence'] < 0.7:
        new_profile.frequency_tolerance = 30.0
        new_profile.harmonic_tolerance = 50.0
    else:
        new_profile.frequency_tolerance = 20.0
        new_profile.harmonic_tolerance = 30.0
    
    # Dodaj profil
    profile_manager.add_profile(new_profile)
    
    # Ustaw jako aktywny
    profile_manager.set_active_profile(len(profile_manager.profiles) - 1)
    
    # Aktualizuj radio buttons
    update_radio_buttons()
    
    # Sugeruj nową nazwę
    num = 1
    while profile_manager.get_profile_by_name(f"Dron_{num}"):
        num += 1
    text_profile_name.set_val(f"Dron_{num}")
    
    print(f"\n✅ PROFIL UTWORZONY I ZAPISANY!")
    print(f"   Nazwa: {profile_name}")
    print(f"   Częstotliwość: {new_profile.target_frequency:.1f} Hz")
    print(f"   Harmonicznych: {len(new_profile.harmonics)}")
    print(f"   Tolerancja: ±{new_profile.frequency_tolerance:.0f} Hz")
    print("="*50 + "\n")

btn_learn_drone.on_clicked(learn_drone_signature)

# PRZYCISK: Usuń profil
ax_delete_profile = plt.axes([0.28, 0.15, 0.12, 0.035])
btn_delete_profile = Button(ax_delete_profile, '🗑️ Usuń profil', color='lightcoral', hovercolor='red')

def delete_profile(event):
    """Usuwa aktywny profil"""
    active_profile = profile_manager.get_active_profile()
    if active_profile:
        if len(profile_manager.profiles) == 1:
            print("❌ Nie można usunąć ostatniego profilu")
            return
        
        name = active_profile.name
        profile_manager.remove_profile(name)
        
        # Ustaw nowy aktywny (pierwszy dostępny)
        if profile_manager.profiles:
            profile_manager.set_active_profile(0)
        
        update_radio_buttons()
        print(f"🗑️ Usunięto profil: {name}")
    else:
        print("❌ Brak aktywnego profilu do usunięcia")

btn_delete_profile.on_clicked(delete_profile)

# PRZYCISK: Eksport profili
ax_export = plt.axes([0.41, 0.15, 0.12, 0.035])
btn_export = Button(ax_export, '📤 Eksport profili', color='lightblue', hovercolor='blue')

def export_profiles(event):
    """Eksportuje profile do pliku JSON"""
    profile_manager.save_profiles()
    print(f"📤 Wyeksportowano {len(profile_manager.profiles)} profili do {profile_manager.filename}")

btn_export.on_clicked(export_profiles)

# PRZYCISK: Import profili
ax_import = plt.axes([0.54, 0.15, 0.12, 0.035])
btn_import = Button(ax_import, '📥 Import profili', color='lightblue', hovercolor='blue')

def import_profiles(event):
    """Importuje profile z pliku JSON"""
    profile_manager.load_profiles()
    update_radio_buttons()
    if profile_manager.profiles:
        profile_manager.set_active_profile(0)
    print(f"📥 Zaimportowano {len(profile_manager.profiles)} profili z {profile_manager.filename}")

btn_import.on_clicked(import_profiles)

# Pozostałe przyciski
ax_reset = plt.axes([0.15, 0.10, 0.08, 0.03])
btn_reset = Button(ax_reset, 'Reset params')

def reset_params(event):
    for slider in sliders.values():
        slider.reset()
    if check.get_status()[0]:
        check.set_active(0)
    if not check.get_status()[1]:
        check.set_active(1)
    params.require_fundamental = True
    params.detect_harmonics = True

btn_reset.on_clicked(reset_params)

ax_clear_all = plt.axes([0.24, 0.10, 0.10, 0.03])
btn_clear_all = Button(ax_clear_all, '🗑️ Usuń wszystkie', color='lightcoral')

def clear_all_profiles(event):
    """Usuwa wszystkie profile (wymaga potwierdzenia)"""
    if len(profile_manager.profiles) > 0:
        profile_manager.profiles = []
        profile_manager.active_profile_index = None
        profile_manager.save_profiles()
        update_radio_buttons()
        print("🗑️ Usunięto wszystkie profile")
    else:
        print("ℹ️ Brak profili do usunięcia")

btn_clear_all.on_clicked(clear_all_profiles)

def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(status)
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]

def update_plot(frame):
    global detection_info, main_title
    
    try:
        # Wykonaj detekcję w zależności od trybu
        if params.detection_mode == 'multi':
            # Detekcja wszystkich profili
            all_results = detector.detect_all_profiles(audio_buffer)
            detection_info['all_results'] = all_results
            detection_info['mode'] = 'multi'
            
            # Sprawdź czy któryś profil został wykryty
            detected_profiles = [r for r in all_results if r['detected']]
            
            if detected_profiles:
                # Pokaż wykryte profile
                profile_names = ', '.join([r['profile_name'] for r in detected_profiles])
                main_title.set_text(f"🚨 WYKRYTO DRONY: {profile_names}")
                main_title.set_color('red')
            else:
                main_title.set_text(f"✓ Brak dronów (monitoruję {len(all_results)} profili)")
                main_title.set_color('green')
            
        else:
            # Detekcja tylko aktywnego profilu
            active_result = detector.detect_active_profile(audio_buffer)
            detection_info['active_result'] = active_result
            detection_info['mode'] = 'single'
            
            if active_result:
                if active_result['detected']:
                    freq_info = f"{active_result['fundamental_freq']:.0f}Hz" if active_result['fundamental_freq'] else "N/A"
                    main_title.set_text(
                        f"🚨 DRON WYKRYTY: {active_result['profile_name']} | "
                        f"Score: {active_result['score']:.2f} | "
                        f"Freq: {freq_info} | "
                        f"Harmonicznych: {active_result['harmonics_found']}"
                    )
                    main_title.set_color('red')
                else:
                    active_profile = profile_manager.get_active_profile()
                    if active_profile:
                        status_text = f"Profil: {active_profile.name} ({active_profile.target_frequency:.0f}Hz)"
                    else:
                        status_text = "Brak aktywnego profilu"
                    main_title.set_text(f"✓ Brak drona | {status_text}")
                    main_title.set_color('green')
            else:
                main_title.set_text("⚠️ Brak aktywnego profilu - wybierz lub utwórz nowy")
                main_title.set_color('orange')
        
        # Spektrogramy
        for idx, (ax, (process_func, title)) in enumerate(zip(ax_specs, processors)):
            ax.clear()
            
            processed_audio = process_func(audio_buffer)
            
            Pxx, freqs, bins, im = ax.specgram(
                processed_audio,
                NFFT=2048,
                Fs=samplerate,
                noverlap=1024,
                cmap='magma'
            )
            
            # Określ zakres Y na podstawie profili
            if profile_manager.profiles:
                max_freq = max([p.target_frequency for p in profile_manager.profiles])
                max_freq_display = min(max_freq * (params.num_harmonics + 2), 5000)
            else:
                max_freq_display = 2000
            
            ax.set_ylim(0, max_freq_display)
            
            # Zaznacz profile
            if idx == 2 or idx == 3:  # Panel "Wszystkie profile" i "Detekcja"
                colors = ['cyan', 'magenta', 'lime', 'yellow', 'orange']
                
                if params.detection_mode == 'multi':
                    # W trybie multi pokazuj wszystkie profile
                    for pi, profile in enumerate(profile_manager.profiles):
                        color = colors[pi % len(colors)]
                        ax.axhline(y=profile.target_frequency, color=color, linestyle='--', 
                                  linewidth=1, alpha=0.6, label=profile.name if idx == 3 else '')
                        
                        # Zaznacz wykryte
                        if 'all_results' in detection_info:
                            for result in detection_info['all_results']:
                                if result['profile_name'] == profile.name and result['detected']:
                                    if result['fundamental_freq']:
                                        ax.axhline(y=result['fundamental_freq'], color=color, 
                                                 linestyle='-', linewidth=2, alpha=0.9)
                else:
                    # W trybie single pokazuj tylko aktywny profil
                    active_profile = profile_manager.get_active_profile()
                    if active_profile:
                        ax.axhline(y=active_profile.target_frequency, color='cyan', linestyle='--', 
                                  linewidth=1.5, alpha=0.7)
                        
                        # Zaznacz harmoniczne
                        if params.detect_harmonics:
                            for n in range(2, params.num_harmonics + 2, 2):
                                harmonic_f = active_profile.target_frequency * n
                                if harmonic_f <= max_freq_display:
                                    ax.axhline(y=harmonic_f, color='yellow', linestyle='--', 
                                             linewidth=0.8, alpha=0.4)
                        
                        # Zaznacz wykryte
                        if detection_info['active_result'] and detection_info['active_result']['detected']:
                            result = detection_info['active_result']
                            if result['fundamental_freq']:
                                ax.axhline(y=result['fundamental_freq'], color='red', 
                                         linestyle='-', linewidth=2, alpha=0.9)
                            
                            for harmonic in result['harmonics_info'][:3]:
                                ax.axhline(y=harmonic['actual'], color='orange', 
                                         linestyle='-', linewidth=1.5, alpha=0.7)
                
                if idx == 3:
                    ax.legend(loc='upper right', fontsize=6, ncol=2)
            
            ax.set_xlabel("Czas [s]", fontsize=7)
            ax.set_ylabel("Freq [Hz]", fontsize=7)
            
            if idx == 3:
                if detection_info['mode'] == 'multi' and 'all_results' in detection_info:
                    detected_count = sum(1 for r in detection_info['all_results'] if r['detected'])
                    details = f"{title}\nWykryte: {detected_count}/{len(detection_info['all_results'])}"
                elif detection_info['active_result']:
                    result = detection_info['active_result']
                    details = (f"{title}\n"
                              f"F:{result['fundamental_score']:.2f} "
                              f"H:{result['harmonics_score']:.2f} "
                              f"A:{result['amplitude_score']:.2f}")
                else:
                    details = title
                ax.set_title(details, fontsize=7)
            else:
                ax.set_title(title, fontsize=8)
            
            ax.tick_params(labelsize=6)
        
        # FFT Plot
        ax_fft.clear()
        
        # Cache FFT
        if frame % 3 == 0:
            fft_data = np.fft.rfft(audio_buffer)
            freqs_fft = np.fft.rfftfreq(len(audio_buffer), 1/samplerate)
            magnitude_fft = np.abs(fft_data)
            update_plot.fft_cache = (freqs_fft, magnitude_fft)
        else:
            if hasattr(update_plot, 'fft_cache'):
                freqs_fft, magnitude_fft = update_plot.fft_cache
            else:
                fft_data = np.fft.rfft(audio_buffer)
                freqs_fft = np.fft.rfftfreq(len(audio_buffer), 1/samplerate)
                magnitude_fft = np.abs(fft_data)
        
        # Określ zakres wykresu
        if profile_manager.profiles:
            max_freq = max([p.target_frequency for p in profile_manager.profiles])
            max_freq_display = min(max_freq * (params.num_harmonics + 2), 5000)
        else:
            max_freq_display = 2000
        
        mask_fft = (freqs_fft >= 0) & (freqs_fft <= max_freq_display)
        
        # Decymacja
        decimation = 5
        freqs_decimated = freqs_fft[mask_fft][::decimation]
        magnitude_decimated = magnitude_fft[mask_fft][::decimation]
        
        ax_fft.semilogy(freqs_decimated, magnitude_decimated, 'b-', 
                        linewidth=0.6, alpha=0.6, label='Spektrum')
        
        # Zaznacz profile
        colors = ['cyan', 'magenta', 'lime', 'yellow', 'orange', 'pink']
        
        if params.detection_mode == 'multi':
            # Pokazuj wszystkie profile
            for pi, profile in enumerate(profile_manager.profiles):
                color = colors[pi % len(colors)]
                
                # Częstotliwość podstawowa
                ax_fft.axvspan(profile.target_frequency - profile.frequency_tolerance,
                              profile.target_frequency + profile.frequency_tolerance,
                              alpha=0.15, color=color)
                ax_fft.axvline(profile.target_frequency, color=color, linestyle='-', 
                              linewidth=1.5, alpha=0.7, label=f"{profile.name}: {profile.target_frequency:.0f}Hz")
                
                # Harmoniczne (tylko pierwsza i druga)
                if params.detect_harmonics:
                    for n in [2, 3]:
                        harmonic_f = profile.target_frequency * n
                        if harmonic_f <= max_freq_display:
                            ax_fft.axvline(harmonic_f, color=color, linestyle='--', 
                                         linewidth=1, alpha=0.4)
            
            # Zaznacz wykryte
            if 'all_results' in detection_info:
                for result in detection_info['all_results']:
                    if result['detected'] and result['fundamental_freq']:
                        # Znajdź kolor profilu
                        for pi, profile in enumerate(profile_manager.profiles):
                            if profile.name == result['profile_name']:
                                color = colors[pi % len(colors)]
                                ax_fft.axvline(result['fundamental_freq'], color=color, 
                                             linestyle='-', linewidth=2.5, alpha=0.9)
                                break
        else:
            # Pokazuj tylko aktywny profil
            active_profile = profile_manager.get_active_profile()
            if active_profile:
                ax_fft.axvspan(active_profile.target_frequency - active_profile.frequency_tolerance,
                              active_profile.target_frequency + active_profile.frequency_tolerance,
                              alpha=0.2, color='cyan', label='Target ±tol')
                ax_fft.axvline(active_profile.target_frequency, color='cyan', linestyle='-', 
                              linewidth=2, alpha=0.8, 
                              label=f"{active_profile.name}: {active_profile.target_frequency:.0f}Hz")
                
                # Harmoniczne
                if params.detect_harmonics:
                    for n in range(2, params.num_harmonics + 2, 2):
                        harmonic_f = active_profile.target_frequency * n
                        if harmonic_f <= max_freq_display:
                            ax_fft.axvline(harmonic_f, color='yellow', linestyle='--', 
                                         linewidth=1.5, alpha=0.6)
                
                # Zaznacz wykryte
                if detection_info['active_result'] and detection_info['active_result']['detected']:
                    result = detection_info['active_result']
                    if result['fundamental_freq']:
                        ax_fft.axvline(result['fundamental_freq'], color='red', linestyle='-', 
                                      linewidth=2.5, alpha=0.9, 
                                      label=f"Wykryty: {result['fundamental_freq']:.1f}Hz")
                    
                    for i, harmonic in enumerate(result['harmonics_info'][:2]):
                        ax_fft.axvline(harmonic['actual'], color='orange', linestyle='-', 
                                      linewidth=2, alpha=0.8)
        
        ax_fft.set_xlabel("Częstotliwość [Hz]", fontsize=10)
        ax_fft.set_ylabel("Amplituda (log)", fontsize=10)
        
        mode_text = "MULTI-PROFILE" if params.detection_mode == 'multi' else "SINGLE-PROFILE"
        ax_fft.set_title(f"FFT Analysis - Tryb: {mode_text}", fontsize=11)
        ax_fft.legend(loc='upper right', fontsize=7, ncol=3)
        ax_fft.grid(True, alpha=0.2, which='both')
        
    except Exception as e:
        print(f"⚠️ Błąd w update_plot: {e}")
    
    return ax_specs + [ax_fft]

update_plot.fft_cache = None

stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate, blocksize=blocksize)

ani = FuncAnimation(fig, update_plot, interval=150, blit=False)

print("🎙️ Uruchamiam MULTI-PROFILE DRONE DETECTOR...")
print(f"\n📂 Wczytano {len(profile_manager.profiles)} profili")
if profile_manager.profiles:
    print("\n📋 DOSTĘPNE PROFILE:")
    for i, profile in enumerate(profile_manager.profiles):
        active = "✓" if i == profile_manager.active_profile_index else " "
        print(f"   {active} {profile.name}: {profile.target_frequency:.1f} Hz "
              f"(harmonicznych: {len(profile.harmonics)}, utworzony: {profile.created_at[:10]})")
else:
    print("\n📋 Brak zapisanych profili - utwórz pierwszy!")

print("\n" + "="*60)
print("💡 INSTRUKCJA:")
print("="*60)
print("\n🎯 TWORZENIE PROFILU:")
print("  1. Wpisz nazwę profilu (np. 'DJI_Mavic', 'Phantom_4')")
print("  2. Gdy słyszysz drona, naciśnij '🎯 JEST DRON (nowy profil)'")
print("  3. System automatycznie stworzy profil z wykrytymi parametrami")
print("  4. Powtórz dla innych dronów!")
print("")
print("🔍 DETEKCJA:")
print("  • SINGLE-PROFILE MODE: Wykrywa tylko wybrany profil")
print("  • MULTI-PROFILE MODE: Wykrywa wszystkie profile jednocześnie!")
print("    (zaznacz checkbox 'Tryb Multi-Profile')")
print("")
print("📊 ZARZĄDZANIE PROFILAMI:")
print("  • Radio buttons - wybierz aktywny profil")
print("  • 🗑️ Usuń profil - usuwa aktywny")
print("  • 🗑️ Usuń wszystkie - czyści całą bazę")
print("  • 📤 Eksport/📥 Import - zapisz/wczytaj profile")
print("")
print("💾 PLIK PROFILI:")
print(f"  Lokalizacja: {profile_manager.filename}")
print("  Format: JSON (można edytować ręcznie)")
print("")
print("🎨 KOLORY NA WYKRESACH:")
print("  W trybie MULTI:")
print("    • Każdy profil ma swój kolor (cyan, magenta, lime, żółty...)")
print("    • Linie przerywane (--) = oczekiwane częstotliwości")
print("    • Linie ciągłe (━) = wykryte częstotliwości")
print("  W trybie SINGLE:")
print("    • Cyan = aktywny profil")
print("    • Czerwony = wykryty dron")
print("    • Pomarańczowy = wykryte harmoniczne")
print("="*60)
print("\n🎤 Nasłuchuję...")
print("CTRL+C aby przerwać...\n")

with stream:
    plt.show()