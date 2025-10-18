import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ======================================================
# CONFIG
# ======================================================
SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 512
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================
# FEATURE EXTRACTION
# ======================================================
def extract_features(filepath):
    """Extract MFCC, FFT magnitude, and spectrogram mean features from an audio file."""
    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)

        # Ensure consistent length (3s)
        max_len = 3 * SAMPLE_RATE
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)))
        else:
            y = y[:max_len]

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Spectrogram
        spec = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
        spec_mean = np.mean(spec, axis=1)

        # FFT (magnitude up to 1000 Hz)
        fft_vals = np.abs(np.fft.rfft(y))
        fft_vals = fft_vals[:1000]
        fft_mean = np.mean(fft_vals)

        # Concatenate all features
        features = np.concatenate([mfcc_mean, spec_mean, [fft_mean]])
        return features
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


# ======================================================
# DATASET
# ======================================================
class FolderAudioDataset(Dataset):
    def __init__(self, root_dir):
        self.filepaths = []
        self.labels = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.label_map = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            folder = os.path.join(root_dir, cls)
            for f in os.listdir(folder):
                if f.endswith(".wav"):
                    self.filepaths.append(os.path.join(folder, f))
                    self.labels.append(self.label_map[cls])

        print(f"Loaded {len(self.filepaths)} files from {len(self.classes)} classes: {self.classes}")

        # Extract all features
        all_features = []
        valid_labels = []
        for fp, label in zip(self.filepaths, self.labels):
            feat = extract_features(fp)
            if feat is not None:
                all_features.append(feat)
                valid_labels.append(label)

        # Normalize
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(all_features)
        self.labels = np.array(valid_labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ======================================================
# MODEL
# ======================================================
class DroneNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ======================================================
# TRAIN LOOP
# ======================================================
def train_model(model, train_loader, val_loader, optimizer, criterion):
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        acc = correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f} | Train Acc: {acc:.4f}")

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = model(x)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total
        print(f"Validation Acc: {val_acc:.4f}\n")


# ======================================================
# MAIN
# ======================================================
def main(dataset_path):
    dataset = FolderAudioDataset(dataset_path)
    X_train, X_val, y_train, y_val = train_test_split(dataset.features, dataset.labels, test_size=0.2, random_state=42)

    train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                              torch.tensor(y_train, dtype=torch.long))
    val_ds = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                            torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train.shape[1]
    num_classes = len(dataset.classes)

    model = DroneNN(input_dim, num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, val_loader, optimizer, criterion)
    torch.save(model.state_dict(), "drone_model.pth")
    print(f"✅ Model saved (classes: {dataset.classes})")


if __name__ == "__main__":
    import sys
    import joblib

    if len(sys.argv) < 2:
        print("Usage: python train_drone_nn.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    dataset = FolderAudioDataset(dataset_path)  # create dataset first

    X_train, X_val, y_train, y_val = train_test_split(
        dataset.features, dataset.labels, test_size=0.2, random_state=42
    )

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train.shape[1]
    num_classes = len(dataset.classes)

    model = DroneNN(input_dim, num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, val_loader, optimizer, criterion)

    torch.save(model.state_dict(), "drone_model.pth")
    joblib.dump(dataset.scaler, "scaler.save")   # ✅ correct reference
    print(f"✅ Model saved (classes: {dataset.classes})")
    print("✅ Scaler saved as scaler.save")
