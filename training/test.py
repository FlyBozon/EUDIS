import torch
import numpy as np
import librosa
import joblib
from train import DroneNN, extract_features

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def predict_audio(model_path, scaler_path, class_names, audio_path):
    model_state = torch.load(model_path, map_location=DEVICE)
    scaler = joblib.load(scaler_path)

    input_dim = len(extract_features(audio_path))
    num_classes = len(class_names)
    model = DroneNN(input_dim, num_classes).to(DEVICE)
    model.load_state_dict(model_state)
    model.eval()

    feats = extract_features(audio_path)
    feats_scaled = scaler.transform([feats])
    x = torch.tensor(feats_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        predicted_class = class_names[pred_idx]
        confidence = probs[pred_idx]

    print(f"🔊 File: {audio_path}")
    print(f"🎯 Predicted: {predicted_class} ({confidence*100:.2f}% confidence)")
    return predicted_class, confidence


if __name__ == "__main__":
    import sys, os
    if len(sys.argv) < 3:
        print("Usage: python test_drone_nn.py <dataset_path> <audio_file>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    audio_path = sys.argv[2]

    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    predict_audio("drone_model.pth", "scaler.save", class_names, audio_path)