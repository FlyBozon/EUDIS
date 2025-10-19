import sounddevice as sd
from scipy.io.wavfile import write

SAMPLE_RATE = 44100
DURATION = 10
OUTPUT_FILE = "recording.wav"

print("🎤 Recording... Speak now!")
audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()
print("✅ Recording complete. Saving file...")

write(OUTPUT_FILE, SAMPLE_RATE, audio_data)
print(f"💾 Saved: {OUTPUT_FILE}")