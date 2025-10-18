import torch
import numpy as np
import pickle
import sys
from scipy.io import wavfile
import librosa

# Import feature extraction functions from training script
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
    D = librosa.stft(data.astype(float), n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    spec_mean = np.mean(magnitude_db, axis=1)
    spec_std = np.std(magnitude_db, axis=1)
    
    return np.concatenate([spec_mean, spec_std])

def extract_mfcc_features(y, sr, n_mfcc=13):
    """Extract MFCC features"""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    mfcc_max = np.max(mfccs, axis=1)
    mfcc_min = np.min(mfccs, axis=1)
    
    return np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])

def extract_all_features(filepath):
    """Extract all features from a WAV file"""
    try:
        samplerate, data = wavfile.read(filepath)
        if len(data.shape) > 1:
            data = data[:, 0]
        
        y, sr = librosa.load(filepath, sr=None, mono=True)
        
        fft_feat = extract_fft_features(data, samplerate)
        spec_feat = extract_spectrogram_features(y, sr)
        mfcc_feat = extract_mfcc_features(y, sr)
        
        features = np.concatenate([fft_feat, spec_feat, mfcc_feat])
        return features
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
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

def predict_audio(wav_path, model, scaler, label_mapping, device):
    """Predict drone detection for a single audio file"""
    # Extract features
    features = extract_all_features(wav_path)
    if features is None:
        return None
    
    # Normalize
    features = scaler.transform(features.reshape(1, -1))
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get class name
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    predicted_class = inv_label_mapping[predicted.item()]
    
    return predicted_class, confidence.item(), probabilities.cpu().numpy()[0]

def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_wav_file>")
        sys.exit(1)
    
    wav_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(wav_path):
        print(f"Error: File {wav_path} not found!")
        sys.exit(1)
    
    # Load model components
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load label mapping
    with open('label_mapping.pkl', 'rb') as f:
        label_mapping = pickle.load(f)
    
    # Load model
    num_classes = len(label_mapping)
    input_size = scaler.n_features_in_
    model = DroneDetectionNN(input_size, num_classes).to(device)
    model.load_state_dict(torch.load('best_drone_model.pth', map_location=device))
    
    print(f"Model loaded successfully on {device}")
    print(f"\nAnalyzing: {wav_path}")
    
    # Predict
    result = predict_audio(wav_path, model, scaler, label_mapping, device)
    
    if result is None:
        print("Failed to process audio file.")
        sys.exit(1)
    
    predicted_class, confidence, all_probs = result
    
    print(f"\n{'='*50}")
    print(f"PREDICTION: {predicted_class.upper()}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"{'='*50}")
    
    print("\nAll class probabilities:")
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    for i, prob in enumerate(all_probs):
        print(f"  {inv_label_mapping[i]}: {prob*100:.2f}%")

if __name__ == "__main__":
    import os
    main()