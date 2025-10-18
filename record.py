import sounddevice as sd
from scipy.io.wavfile import write

# === SETTINGS ===
SAMPLE_RATE = 44100  # Sample rate in Hz
DURATION = 360        # Duration to record in seconds
OUTPUT_FILE = "recording.wav"

print("🎤 Recording... Speak now!")
audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()  # Wait until recording is finished
print("✅ Recording complete. Saving file...")

write(OUTPUT_FILE, SAMPLE_RATE, audio_data)
print(f"💾 Saved: {OUTPUT_FILE}")
