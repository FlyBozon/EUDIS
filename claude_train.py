import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm

# ======================== FEATURE EXTRACTION ========================

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
    
    # Bin the FFT into n_bins for consistent feature size
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
    # Compute STFT
    D = librosa.stft(data.astype(float), n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    
    # Convert to dB scale
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # Aggregate: mean and std across time for each frequency bin
    spec_mean = np.mean(magnitude_db, axis=1)
    spec_std = np.std(magnitude_db, axis=1)
    
    return np.concatenate([spec_mean, spec_std])

def extract_mfcc_features(y, sr, n_mfcc=13):
    """Extract MFCC features"""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Statistical aggregation across time
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    mfcc_max = np.max(mfccs, axis=1)
    mfcc_min = np.min(mfccs, axis=1)
    
    return np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])

def extract_all_features(filepath):
    """Extract all features from a WAV file"""
    try:
        # Load with scipy for FFT and spectrogram
        samplerate, data = wavfile.read(filepath)
        if len(data.shape) > 1:  # stereo -> mono
            data = data[:, 0]
        
        # Load with librosa for MFCC
        y, sr = librosa.load(filepath, sr=None, mono=True)
        
        # Extract features
        fft_feat = extract_fft_features(data, samplerate)
        spec_feat = extract_spectrogram_features(y, sr)
        mfcc_feat = extract_mfcc_features(y, sr)
        
        # Concatenate all features
        features = np.concatenate([fft_feat, spec_feat, mfcc_feat])
        return features
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

# ======================== DATASET CLASS ========================

class DroneAudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ======================== NEURAL NETWORK ========================

class DroneDetectionNN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes=[512, 256, 128]):
        super(DroneDetectionNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ======================== TRAINING FUNCTIONS ========================

def load_dataset(data_dir, label_mapping):
    """
    Load dataset from directory structure:
    data_dir/
        class1/
            file1.wav
            file2.wav
        class2/
            file1.wav
    
    label_mapping: dict like {'drone': 0, 'no_drone': 1} or {'type1': 0, 'type2': 1, 'unknown': 2}
    """
    features_list = []
    labels_list = []
    
    for class_name, label in label_mapping.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found!")
            continue
        
        wav_files = [f for f in os.listdir(class_dir) if f.lower().endswith('.wav')]
        print(f"\nProcessing {len(wav_files)} files from class '{class_name}'...")
        
        for wav_file in tqdm(wav_files):
            filepath = os.path.join(class_dir, wav_file)
            features = extract_all_features(filepath)
            
            if features is not None:
                features_list.append(features)
                labels_list.append(label)
    
    return np.array(features_list), np.array(labels_list)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Train the model"""
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_drone_model.pth')
            print(f"✓ Saved best model with accuracy: {best_val_acc:.2f}%")
    
    return train_losses, val_accuracies

# ======================== MAIN TRAINING SCRIPT ========================

def main():
    # Configuration
    BINARY_MODE = True  # Set to False for multi-class classification
    
    if BINARY_MODE:
        DATA_DIR = 'DroneAudioDataset/Binary_Drone_Audio'  # Contains 'drone' and 'no_drone' folders
        LABEL_MAPPING = {'drone': 0, 'no_drone': 1}
        NUM_CLASSES = 2
    else:
        DATA_DIR = 'dataset_multiclass'  # Contains 'type1', 'type2', 'unknown' folders
        LABEL_MAPPING = {'type1': 0, 'type2': 1, 'unknown': 2}
        NUM_CLASSES = 3
    
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and extract features
    print("Loading dataset and extracting features...")
    X, y = load_dataset(DATA_DIR, LABEL_MAPPING)
    print(f"\nDataset loaded: {len(X)} samples, {X.shape[1]} features per sample")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split dataset
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=42, stratify=y_temp
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Save scaler for inference
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Saved scaler to scaler.pkl")
    
    # Create datasets and dataloaders
    train_dataset = DroneAudioDataset(X_train, y_train)
    val_dataset = DroneAudioDataset(X_val, y_val)
    test_dataset = DroneAudioDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    input_size = X_train.shape[1]
    model = DroneDetectionNN(input_size, NUM_CLASSES).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print("\nStarting training...")
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device
    )
    
    # Test final model
    model.load_state_dict(torch.load('best_drone_model.pth'))
    model.eval()
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100 * correct / total
    print(f"\n{'='*50}")
    print(f"FINAL TEST ACCURACY: {test_acc:.2f}%")
    print(f"{'='*50}")
    
    # Save label mapping
    with open('label_mapping.pkl', 'wb') as f:
        pickle.dump(LABEL_MAPPING, f)
    print("✓ Saved label mapping to label_mapping.pkl")
    
    print("\nTraining complete! Files saved:")
    print("  - best_drone_model.pth (model weights)")
    print("  - scaler.pkl (feature scaler)")
    print("  - label_mapping.pkl (class labels)")

if __name__ == "__main__":
    main()