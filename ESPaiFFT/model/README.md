# Model dla ESP32-S3 (TFLite INT8)

Ten katalog zawiera narzędzia do przygotowania lekkiego modelu do uruchomienia na ESP32-S3 (8MB PSRAM).

## Wejście/wyjście modelu
- Wejście: Mel spektrogram 96x96x1 (zgodny z `MelSpectrogramGenerator` w `include/mel_spectrogram.h`)
- Wyjście: 2 klasy (drone / other)

## Jak zbudować i wgrać lekki model
1. (Opcjonalnie) Zainstaluj TensorFlow dla Pythona 3.9+.
2. Uruchom skrypt:
   - `convert_model_tiny.py` tworzy mały model DS-CNN i konwertuje go do TFLite INT8.
   - Skrypt dodatkowo generuje nagłówek C `drone_model.h` w tym katalogu.
3. Skopiuj `drone_model.h` do `../include/`:
   - `ESPaiFFT/include/drone_model.h`
4. Włącz model w kompilacji: w `ESPaiFFT/platformio.ini` zmień `-DUSE_DRONE_MODEL=0` na `-DUSE_DRONE_MODEL=1`.
5. Zbuduj i wgraj projekt (zadanie PlatformIO: Build & Upload).

## Uwaga dot. jakości
Skrypt zawiera szkic architektury bez treningu. Aby uzyskać sensowne wyniki:
- Wytrenuj model w Keras na własnym zbiorze danych.
- Zastąp inicjalizację wag (`build_tiny_dscnn`) wczytaniem wytrenowanego modelu (`tf.keras.models.load_model`).
- Upewnij się, że wejście ma kształt 96x96x1 i że cała konwersja jest INT8 (pełna kwantyzacja).

## Rozmiar i pamięć
- Oczekiwany rozmiar modelu TFLite: ~50-300 KB (zależnie od architektury).
- Tensor Arena (w `drone_detector.h`): startowo 200 KB; dostosuj w razie błędów `MicroInterpreter`.
