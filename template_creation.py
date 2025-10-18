import os
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import librosa
from scipy.spatial.distance import cdist
import time

SR = 44100
BLOCK = 1024
WINDOW_SEC = 1.0
OUTPUT_DIR = "detections"
MARGIN = 0.1  # threshold margin dron vs tło

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_templates_from_folder(folder_path):
    templates = []
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".wav"):
            continue
        path = os.path.join(folder_path, fname)
        y, _ = librosa.load(path, sr=SR)
        n_window = int(SR * WINDOW_SEC)
        for start in range(0, len(y)-n_window, n_window//2):  # 50% overlap
            win = y[start:start+n_window]
            mfcc = librosa.feature.mfcc(win, sr=SR, n_mfcc=13)
            templates.append(mfcc)
    return templates

def match_template(mfcc_window, templates):
    sims = []
    for t in templates:
        sims.append(1 - cdist(mfcc_window.flatten()[None,:], t.flatten()[None,:], metric='cosine')[0,0])
    return max(sims)

print("Ładowanie template'ów dronów i tła...")
drone_templates = extract_templates_from_folder("dataset/drones")
background_templates = extract_templates_from_folder("dataset/not_drones")
print(f"Ładowano {len(drone_templates)} template'ów dronów i {len(background_templates)} tła.")

def live_drone_detector():
    buffer = np.zeros(int(SR * 5))
    detections = []

    # przygotuj wykres
    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_ylim(0, SR//2)
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Częstotliwość [Hz]")
    ax.set_title("Spektrogram na żywo + detekcja drona")
    detected_text = ax.text(0.02,0.95,"", transform=ax.transAxes, color='red', fontsize=14, fontweight='bold')

    # callback dźwięku
    def audio_callback(indata, frames, time_info, status):
        nonlocal buffer
        buffer = np.roll(buffer, -frames)
        buffer[-frames:] = indata[:,0]

    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SR, blocksize=BLOCK)

    def update(frame):
        sig = buffer[-int(WINDOW_SEC*SR):]
        if np.all(sig==0):
            return

        # MFCC aktualnego okna
        mfcc_win = librosa.feature.mfcc(sig, sr=SR, n_mfcc=13)

        # porównanie z template
        score_drone = match_template(mfcc_win, drone_templates)
        score_bg = match_template(mfcc_win, background_templates)

        # decyzja
        if score_drone > score_bg + MARGIN:
            msg = f"🚁 DRON wykryty! score_drone={score_drone:.3f} score_bg={score_bg:.3f}"
            detected_text.set_text(msg)
            detected_text.set_color('red')
            print(msg)

            # zapis spektrogramu
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plt.specgram(sig, NFFT=1024, Fs=SR, noverlap=512, cmap='magma')
            plt.savefig(os.path.join(OUTPUT_DIR,f"detection_{timestamp}.png"), dpi=200)
            detections.append((timestamp, score_drone))
        else:
            detected_text.set_text("Brak drona")
            detected_text.set_color('green')

        # spektrogram live
        ax.cla()
        ax.specgram(sig, NFFT=1024, Fs=SR, noverlap=512, cmap='magma')
        ax.set_ylim(0, SR//2)
        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Częstotliwość [Hz]")
        ax.set_title("Spektrogram na żywo + detekcja drona")
        ax.text(0.02,0.95, detected_text.get_text(), transform=ax.transAxes,
                color=detected_text.get_color(), fontsize=14, fontweight='bold')

    ani = FuncAnimation(fig, update, interval=500, blit=False)
    print("🎙️ Tryb detekcji na żywo... CTRL+C aby zakończyć.")
    with stream:
        plt.show()

    # zapis wykryć
    if detections:
        txt_path = os.path.join(OUTPUT_DIR,"detections.txt")
        with open(txt_path,"w") as f:
            for t,s in detections:
                f.write(f"{t}\t{s:.3f}\n")
        print(f"Zapisano listę wykryć: {txt_path}")

if __name__ == "__main__":
    live_drone_detector()
