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
import json
from datetime import datetime
from tqdm import tqdm
import cv2
import warnings
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
warnings.filterwarnings('ignore')


def create_spectrogram_image(filepath, target_size=(224, 224), save_visualization=False):
    """
    Create spectrogram as IMAGE (like a photo) for CNN.
    This treats drone detection as pure computer vision problem.
    """
    try:
        y, sr = librosa.load(filepath, sr=22050, mono=True)

        nyquist = sr / 2
        low = 50 / nyquist
        high = min(1000 / nyquist, 0.99)
        b, a = signal.butter(5, [low, high], btype='band')
        y_filtered = signal.filtfilt(b, a, y)

        mel_spec = librosa.feature.melspectrogram(
            y=y_filtered,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            fmin=50,
            fmax=1000
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        mel_spec_norm = ((mel_spec_db - mel_spec_db.min()) /
                        (mel_spec_db.max() - mel_spec_db.min()) * 255).astype(np.uint8)

        spectrogram_image = cv2.resize(mel_spec_norm, target_size, interpolation=cv2.INTER_LINEAR)

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

        nyquist = sr / 2
        low = 50 / nyquist
        high = min(1000 / nyquist, 0.99)
        b, a = signal.butter(5, [low, high], btype='band')
        y_filtered = signal.filtfilt(b, a, y)

        mel_spec = librosa.feature.melspectrogram(y=y_filtered, sr=sr, n_mels=target_size[0], fmax=1000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        chroma = librosa.feature.chroma_cqt(y=y_filtered, sr=sr)
        chroma = cv2.resize(chroma, (mel_spec_db.shape[1], target_size[0]))

        contrast = librosa.feature.spectral_contrast(y=y_filtered, sr=sr)
        contrast = cv2.resize(contrast, (mel_spec_db.shape[1], target_size[0]))

        def normalize(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        mel_norm = normalize(mel_spec_db)
        chroma_norm = normalize(chroma)
        contrast_norm = normalize(contrast)

        mel_resized = cv2.resize(mel_norm, target_size)
        chroma_resized = cv2.resize(chroma_norm, target_size)
        contrast_resized = cv2.resize(contrast_norm, target_size)

        multi_channel = np.stack([mel_resized, chroma_resized, contrast_resized], axis=-1)

        return multi_channel

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


class SpectrogramImageDataset(Dataset):
    def __init__(self, images, labels, use_multi_channel=False):
        """
        Dataset that treats spectrograms as images
        """
        self.use_multi_channel = use_multi_channel

        if use_multi_channel:
            self.images = torch.FloatTensor(images).permute(0, 3, 1, 2)
        else:
            self.images = torch.FloatTensor(images).unsqueeze(1)

        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class SimpleDroneCNN(nn.Module):
    """
    Simple CNN inspired by VGG architecture for spectrogram classification
    """
    def __init__(self, num_classes, input_channels=1):
        super(SimpleDroneCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

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


def load_spectrogram_dataset(data_dir, label_mapping, target_size=(224, 224),
                             use_multi_channel=False, save_examples=False):
    """
    Load dataset and convert to spectrogram images
    """
    images_list = []
    labels_list = []

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


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device, scheduler=None, model_save_path='best_drone_visual_model.pth'):
    """Train the visual model"""
    best_val_f1 = 0.0
    patience = 20
    patience_counter = 0

    for epoch in range(num_epochs):
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

        model.eval()
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = 100 * val_correct / val_total

        _, _, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='macro')

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")

        if scheduler:
            scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"  ✓ Saved best model: F1 = {best_val_f1:.4f} -> {model_save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return best_val_f1


def main():
    BINARY_MODE = True
    USE_MULTI_CHANNEL = False
    MODEL_TYPE = 'resnet'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if BINARY_MODE:
        DATA_DIR = os.path.join(script_dir, '..', 'DroneAudioDataset', 'Binary_Drone_Audio')
        LABEL_MAPPING = {'drone': 0, 'unknown': 1}
        NUM_CLASSES = 2
    else:
        DATA_DIR = 'dataset_multiclass'
        LABEL_MAPPING = {'type1': 0, 'type2': 1, 'unknown': 2}
        NUM_CLASSES = 3

    TARGET_SIZE = (224, 224)
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Channels: {'3 (Multi-channel)' if USE_MULTI_CHANNEL else '1 (Grayscale)'}")

    print("\nConverting audio to spectrogram images...")
    images, labels = load_spectrogram_dataset(
        DATA_DIR, LABEL_MAPPING, TARGET_SIZE,
        use_multi_channel=USE_MULTI_CHANNEL,
        save_examples=True
    )

    print(f"\nDataset loaded: {len(images)} samples")
    if len(images) == 0:
        print("\nERROR: No samples found in dataset.")
        print(f"  Expected dataset directory (binary mode): {DATA_DIR}")
        print("  Make sure the dataset files are present and that class subfolders exist (e.g. 'drone', 'unknown').")
        print("  You can also change working directory or adjust DATA_DIR in the script.")
        return

    print(f"Image shape: {images[0].shape}")
    print(f"Class distribution: {np.bincount(labels)}")

    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=TEST_SIZE, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=42, stratify=y_temp
    )

    train_dataset = SpectrogramImageDataset(X_train, y_train, USE_MULTI_CHANNEL)
    val_dataset = SpectrogramImageDataset(X_val, y_val, USE_MULTI_CHANNEL)
    test_dataset = SpectrogramImageDataset(X_test, y_test, USE_MULTI_CHANNEL)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_channels = 3 if USE_MULTI_CHANNEL else 1

    if MODEL_TYPE == 'resnet':
        model = DroneResNet(NUM_CLASSES, input_channels).to(device)
    else:
        model = SimpleDroneCNN(NUM_CLASSES, input_channels).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'best_drone_visual_model_{ts}.pth'
    config_filename = f'visual_model_config_{ts}.pkl'
    labelmap_filename = f'label_mapping_{ts}.pkl'
    metrics_filename = f'visual_metrics_{ts}.json'
    cm_image_filename = f'confusion_matrix_{ts}.png'

    print("\nStarting training...")
    best_f1 = train_model(model, train_loader, val_loader, criterion,
                           optimizer, NUM_EPOCHS, device, scheduler, model_save_path=model_filename)

    model.load_state_dict(torch.load(model_filename))
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100 * correct / total
    print(f"\n{'='*60}")
    print(f"FINAL TEST ACCURACY: {test_acc:.2f}%")
    print(f"{'='*60}")

    print("\nDetailed Classification Metrics:")
    print("=" * 60)

    target_names = list(LABEL_MAPPING.keys())
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    metrics = {
        'timestamp': ts,
        'test_accuracy': float(test_acc),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
    }

    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None)
    metrics['per_class'] = {}
    for i, name in enumerate(target_names):
        metrics['per_class'][name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    metrics['macro'] = {'precision': float(macro_precision), 'recall': float(macro_recall), 'f1': float(macro_f1)}
    metrics['weighted'] = {'precision': float(weighted_precision), 'recall': float(weighted_recall), 'f1': float(weighted_f1)}

    with open(metrics_filename, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    try:
        plt.figure(figsize=(6,6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
        plt.tight_layout()
        plt.savefig(cm_image_filename, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Failed to save confusion matrix image: {e}")

    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None)
    print(f"\nPer-class Precision: {precision}")
    print(f"Per-class Recall: {recall}")
    print(f"Per-class F1-Score: {f1}")
    print(f"Support: {support}")

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    print(f"\nMacro Average - Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")
    print(f"Weighted Average - Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1: {weighted_f1:.4f}")

    config = {
        'model_type': MODEL_TYPE,
        'use_multi_channel': USE_MULTI_CHANNEL,
        'target_size': TARGET_SIZE,
        'input_channels': input_channels,
        'num_classes': NUM_CLASSES
    }

    with open(config_filename, 'wb') as f:
        pickle.dump(config, f)

    with open(labelmap_filename, 'wb') as f:
        pickle.dump(LABEL_MAPPING, f)

    print("\nFiles saved:")
    print("  - best_drone_visual_model.pth")
    print("  - visual_model_config.pkl")
    print("  - label_mapping.pkl")
    print("  - *_spectrogram.png (first 5 examples)")

if __name__ == "__main__":
    main()