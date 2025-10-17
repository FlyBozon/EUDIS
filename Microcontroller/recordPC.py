import serial
import wave
import struct
import time

SAMPLE_RATE = 44100
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
RECORD_SECONDS = 10
GAIN = 20.0

ser = serial.Serial('COM7', 921600, timeout=1)

wav_file = wave.open('recording.wav', 'wb')
wav_file.setnchannels(CHANNELS)
wav_file.setsampwidth(SAMPLE_WIDTH)
wav_file.setframerate(SAMPLE_RATE)

print(f"Recording for {RECORD_SECONDS}")

start_time = time.time()

try:
    while time.time() - start_time < RECORD_SECONDS:
        byte1 = ser.read(1)
        if byte1 == b'\xAA':
            byte2 = ser.read(1)
            if byte2 == b'\x55':
                count_bytes = ser.read(2)
                if len(count_bytes) < 2:
                    continue
                sample_count = struct.unpack('<H', count_bytes)[0]

                audio_data = ser.read(sample_count * 2)
                if len(audio_data) < sample_count * 2:
                    continue

                samples = struct.unpack('<' + 'h' * sample_count, audio_data)
                amplified_samples = []
                for s in samples:
                    amplified = int(s * GAIN)
                    if amplified > 32767:
                        amplified = 32767
                    elif amplified < -32768:
                        amplified = -32768
                    amplified_samples.append(amplified)
                amplified_data = struct.pack('<' + 'h' * sample_count, *amplified_samples)

                wav_file.writeframes(amplified_data)
finally:
    wav_file.close()
    ser.close()
    print("Recording complete. File saved as recording.wav")
