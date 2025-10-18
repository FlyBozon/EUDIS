import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
from scipy import signal
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
import cv2
import warnings
warnings.filterwarnings('ignore')

# ======================== SPECTROGRAM GENERATION ========================

def create_spectrogram_image(filepath, target_size=(224, 224), save_visualization=False):
    """
    Create spectrogram as IMAGE (like a photo) for CNN.
    This treats drone detection as pure computer vision problem.
    """
    try:
        # Load audio
        y, sr = librosa.load(filepath, sr=22050, mono=True)  # Lower SR for faster processing
        
        # Apply bandpass filter (50-1000 Hz for drones)
        nyquist = sr / 2
        low = 50 / nyquist
        high = min(1000 / nyquist, 0.99)
        b, a = signal.butter(5, [low, high], btype='band')
        y_filtered = signal.filtfilt(b, a, y)
        
        # Create mel spectrogram (better for visualization)
        mel_spec = librosa.feature.melspectrogram(
            y=y_filtered, 
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            fmin=50,
            fmax=1000
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to 0-255 (like an image)
        mel_spec_norm = ((mel_spec_db - mel_spec_db.min()) / 
                        (mel_spec_db.max() - mel_spec_db.min()) * 255).astype(np.uint8)
        
        # Resize to target size using CV2 (like image preprocessing)
        spectrogram_image = cv2.resize(mel_spec_norm, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Optional: Save visualization to see what model sees
        if save_visualization:
            output_path = filepath.replace('.wav', '_spectrogram.png')
            plt.figure(figsize=(10, 4))
            plt.imshow(spectrogram_image, aspect='auto', origin='lower', cmap='magma')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram: {os.path.basename(filepath)}')
            plt.xlabel('Time')
            plt.ylabel('Frequency (Hz)')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        return spectrogram_image
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def create_multi_representation(filepath, target_size=(224, 224)):
    """
    Create 3-channel image combining different audio representations.
    Like RGB channels but for audio: [Mel Spectrogram, Chromagram, Spectral Contrast]
    """
    try:
        y, sr = librosa.load(filepath, sr=22050, mono=True)
        
        # Apply filter
        nyquist = sr / 2
        low = 50 / nyquist
        high = min(1000 / nyquist, 0.99)
        b, a = signal.butter(5, [low, high], btype='band')
        y_filtered = signal.filtfilt(b, a, y)
        
        # Channel 1: Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y_filtered, sr=sr, n_mels=target_size[0], fmax=1000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Channel 2: Chromagram (harmonic content)
        chroma = librosa.feature.chroma_cqt(y=y_filtered, sr=sr)
        chroma = cv2.resize(chroma, (mel_spec_db.shape[1], target_size[0]))
        
        # Channel 3: Spectral Contrast (texture)
        contrast = librosa.feature.spectral_contrast(y=y_filtered, sr=sr)
        contrast = cv2.resize(contrast, (mel_spec_db.shape[1], target_size[0]))
        
        # Normalize each channel to 0-1
        def normalize(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        mel_norm = normalize(mel_spec_db)
        chroma_norm = normalize(chroma)
        contrast_norm = normalize(contrast)
        
        # Resize to target
        mel_resized = cv2.resize(mel_norm, target_size)
        chroma_resized = cv2.resize(chroma_norm, target_size)
        contrast_resized = cv2.resize(contrast_norm, target_size)
        
        # Stack as 3-channel image (H, W, 3)
        multi_channel = np.stack([mel_resized, chroma_resized, contrast_resized], axis=-1)
        
        return multi_channel
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

# ======================== DATASET CLASS ========================

class SpectrogramImageDataset(Dataset):
    def __init__(self, images, labels, use_multi_channel=False):
        """
        Dataset that treats spectrograms as images
        """
        self.use_multi_channel = use_multi_channel
        
        if use_multi_channel:
            # (N, H, W, 3) -> (N, 3, H, W) for PyTorch
            self.images = torch.FloatTensor(images).permute(0, 3, 1, 2)
        else:
            # (N, H, W) -> (N, 1, H, W) for PyTorch (grayscale)
            self.images = torch.FloatTensor(images).unsqueeze(1)
        
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# ======================== CNN MODELS ========================

class SimpleDroneCNN(nn.Module):
    """
    Simple CNN inspired by VGG architecture for spectrogram classification
    """
    def __init__(self, num_classes, input_channels=1):
        super(SimpleDroneCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class DroneResNet(nn.Module):
    """
    ResNet-like architecture for drone detection from spectrograms
    Better for learning complex patterns
    """
    def __init__(self, num_classes, input_channels=1):
        super(DroneResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# ======================== DATA LOADING ========================

def load_spectrogram_dataset(data_dir, label_mapping, target_size=(224, 224), 
                             use_multi_channel=False, save_examples=False):
    """
    Load dataset and convert to spectrogram images
    """
    images_list = []
    labels_list = []
    
    # Save a few examples
    examples_saved = 0
    max_examples = 5
    
    for class_name, label in label_mapping.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found!")
            continue
        
        wav_files = [f for f in os.listdir(class_dir) if f.lower().endswith('.wav')]
        print(f"\nProcessing {len(wav_files)} files from class '{class_name}'...")
        
        for wav_file in tqdm(wav_files):
            filepath = os.path.join(class_dir, wav_file)
            
            # Save visualization for first few examples
            save_viz = save_examples and examples_saved < max_examples
            
            if use_multi_channel:
                image = create_multi_representation(filepath, target_size)
            else:
                image = create_spectrogram_image(filepath, target_size, save_visualization=save_viz)
            
            if image is not None:
                images_list.append(image)
                labels_list.append(label)
                
                if save_viz:
                    examples_saved += 1
    
    return np.array(images_list), np.array(labels_list)

# ======================== TRAINING ========================

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, scheduler=None):
    """Train the visual model"""
    best_val_acc = 0.0
    patience = 20
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_drone_visual_model.pth')
            print(f"  ✓ Saved best model: {best_val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return best_val_acc

# ======================== MAIN ========================

def main():
    # Configuration
    BINARY_MODE = True
    USE_MULTI_CHANNEL = False  # Set True for 3-channel representation
    MODEL_TYPE = 'resnet'  # 'simple' or 'resnet'
    
    if BINARY_MODE:
        DATA_DIR = 'DroneAudioDataset/Binary_Drone_Audio'
        LABEL_MAPPING = {'drone': 0, 'unknown': 1}
        NUM_CLASSES = 2
    else:
        DATA_DIR = 'dataset_multiclass'
        LABEL_MAPPING = {'type1': 0, 'type2': 1, 'unknown': 2}
        NUM_CLASSES = 3
    
    # Hyperparameters
    TARGET_SIZE = (224, 224)  # Standard image size
    BATCH_SIZE = 16
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.001
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Channels: {'3 (Multi-channel)' if USE_MULTI_CHANNEL else '1 (Grayscale)'}")
    
    # Load dataset as images
    print("\nConverting audio to spectrogram images...")
    images, labels = load_spectrogram_dataset(
        DATA_DIR, LABEL_MAPPING, TARGET_SIZE, 
        use_multi_channel=USE_MULTI_CHANNEL,
        save_examples=True  # Save first 5 examples
    )
    
    print(f"\nDataset loaded: {len(images)} samples")
    print(f"Image shape: {images[0].shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Split dataset
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=TEST_SIZE, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=42, stratify=y_temp
    )
    
    # Create datasets
    train_dataset = SpectrogramImageDataset(X_train, y_train, USE_MULTI_CHANNEL)
    val_dataset = SpectrogramImageDataset(X_val, y_val, USE_MULTI_CHANNEL)
    test_dataset = SpectrogramImageDataset(X_test, y_test, USE_MULTI_CHANNEL)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    input_channels = 3 if USE_MULTI_CHANNEL else 1
    
    if MODEL_TYPE == 'resnet':
        model = DroneResNet(NUM_CLASSES, input_channels).to(device)
    else:
        model = SimpleDroneCNN(NUM_CLASSES, input_channels).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Train
    print("\nStarting training...")
    best_acc = train_model(model, train_loader, val_loader, criterion, 
                           optimizer, NUM_EPOCHS, device, scheduler)
    
    # Test
    model.load_state_dict(torch.load('best_drone_visual_model.pth'))
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    print(f"\n{'='*60}")
    print(f"FINAL TEST ACCURACY: {test_acc:.2f}%")
    print(f"{'='*60}")
    
    # Save metadata
    config = {
        'model_type': MODEL_TYPE,
        'use_multi_channel': USE_MULTI_CHANNEL,
        'target_size': TARGET_SIZE,
        'input_channels': input_channels,
        'num_classes': NUM_CLASSES
    }
    
    with open('visual_model_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    with open('label_mapping.pkl', 'wb') as f:
        pickle.dump(LABEL_MAPPING, f)
    
    print("\nFiles saved:")
    print("  - best_drone_visual_model.pth")
    print("  - visual_model_config.pkl")
    print("  - label_mapping.pkl")
    print("  - *_spectrogram.png (first 5 examples)")

if __name__ == "__main__":
    main()