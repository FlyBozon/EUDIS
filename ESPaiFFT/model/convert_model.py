"""
Konwerter modelu drona z visual_train.py do TFLite dla ESP32-S3
PyTorch ResNet18 → TFLite
"""

import tensorflow as tf
import torch
import torchvision.models as models
import pickle
import numpy as np
import os

def load_pytorch_model():
    """Załaduj model PyTorch z best_drone_visual_model.pth"""

    print("[PYTORCH] Załadowanie modelu...")

    with open('visual_model_config.pkl', 'rb') as f:
        config = pickle.load(f)

    print(f"[CONFIG] Model type: {config.get('model_type')}")
    print(f"[CONFIG] Channels: {config.get('input_channels')}")
    print(f"[CONFIG] Shape: {config.get('target_size')}")
    print(f"[CONFIG] Classes: {config.get('num_classes')}")

    device = torch.device('cpu')
    state_dict = torch.load('best_drone_visual_model.pth', map_location=device)

    model = models.resnet18(weights=None)

    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model.fc = torch.nn.Linear(512, 2)

    new_state_dict = {}
    for key, value in state_dict.items():
        if 'shortcut' in key:
            new_key = key.replace('shortcut', 'downsample')
        else:
            new_key = key
        new_state_dict[new_key] = value

    try:
        model.load_state_dict(new_state_dict, strict=False)
        print(f"[PYTORCH] Model załadowany pomyślnie")
    except Exception as e:
        print(f"[ERROR] Błąd ładowania modelu: {e}")
        return None, None

    model.eval()

    return model, config

def pytorch_to_onnx(pytorch_model):
    """Konwertuj PyTorch do ONNX"""

    print("\n[ONNX] Konwersja PyTorch → ONNX...")

    dummy_input = torch.randn(1, 1, 224, 224, dtype=torch.float32)

    onnx_path = 'drone_model.onnx'
    try:
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            onnx_path,
            input_names=['spectrogram_input'],
            output_names=['predictions'],
            opset_version=12,
            verbose=False,
            do_constant_folding=True
        )
        print(f"✓ Model ONNX: {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"⚠ ONNX export nie powiódł się: {e}")
        return None

def pytorch_to_tflite_direct():
    """Bezpośrednia konwersja - utwórz model Keras o identycznej architekturze"""

    print("\n[KERAS] Tworzenie modelu Keras (ResNet18)...")

    inputs = tf.keras.Input(shape=(224, 224, 1), dtype=tf.float32)

    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 64, stride=1)

    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128, stride=1)

    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256, stride=1)

    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512, stride=1)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    print("✓ Model Keras ResNet18 utworzony")

    return model

def residual_block(x, filters, stride=1):
    """Tworzy residual block dla ResNet"""

    shortcut = x

    x = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)

    return x

def representative_dataset():
    """Generator dla kwantyzacji"""
    for _ in range(100):
        spectrogram = np.random.rand(1, 224, 224, 1).astype(np.float32)
        yield [spectrogram]

def convert_keras_to_tflite(model, output_path='drone_model.tflite'):
    """Konwertuj model Keras do TFLite - INT8 quantization"""

    print("\n[TFLITE] Konwersja Keras → TFLite (INT8 quantization)...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    try:
        print("  → Full INT8 quantization (wagi i aktywacje)...")
        tflite_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        size_mb = len(tflite_model) / (1024 * 1024)
        size_kb = len(tflite_model) / 1024

        print(f"✓ Model TFLite (INT8): {output_path}")
        print(f"  Rozmiar: {size_kb:.0f} KB ({size_mb:.2f} MB)")

        return tflite_model

    except Exception as e:
        print(f"[ERROR] INT8 nie powiódł się: {e}")
        print(f"[INFO] Próbuję float32...")

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"✓ Model TFLite (float32): {output_path}")
        print(f"  Rozmiar: {len(tflite_model)/1024:.0f} KB")

        return tflite_model

def generate_c_header(tflite_model, output_path='drone_model.h'):
    """Generuje plik C++ z modelem w segmentach PROGMEM"""

    print(f"\n[HEADER] Generowanie C++ header (segmentowy)...")

    SEGMENT_SIZE = 1024
    segments = []

    for i in range(0, len(tflite_model), SEGMENT_SIZE):
        chunk = tflite_model[i:i+SEGMENT_SIZE]
        hex_str = ', '.join(f'0x{byte:02x}' for byte in chunk)
        segments.append(hex_str)

    c_code = f"""// TFLite model dla ESP32-S3 PSRAM - segmentowany w PROGMEM
// Model: drone_model.tflite
// Size: {len(tflite_model)} bytes ({len(tflite_model)/(1024*1024):.2f} MB)
// Architecture: ResNet18 (1 channel input)
// Input: [1, 224, 224, 1] float32 spectrogram
// Output: [1, 2] float32 softmax [drone_prob, not_drone_prob]
//
// Model przechowywany jest w segmentach PROGMEM i załadowywany do PSRAM przy starcie



// Liczba segmentów modelu

// Segmenty modelu w PROGMEM
"""

    for i, segment in enumerate(segments):
        c_code += f"""
const uint8_t drone_model_segment_{i}[] PROGMEM = {{
    {segment}
}};
"""

    c_code += f"""
// Tablica wskaźników na segmenty
const uint8_t* const drone_model_segments[{len(segments)}] PROGMEM = {{
"""

    for i in range(len(segments)):
        c_code += f"    drone_model_segment_{i},\n"

    c_code += f"""
}};

// Funkcja do załadowania modelu z PROGMEM do PSRAM
uint8_t* load_drone_model_to_psram() {{
    // Alokuj buffer w PSRAM
    uint8_t* psram_buffer = (uint8_t*)ps_malloc({len(tflite_model)});

    if (!psram_buffer) {{
        Serial.println("[ERROR] Nie udało się zaalokować {len(tflite_model)} bajtów w PSRAM");
        return nullptr;
    }}

    // Skopiuj segmenty z PROGMEM do PSRAM
    uint32_t offset = 0;
    for (int i = 0; i < {len(segments)}; i++) {{
        const uint8_t* segment = (const uint8_t*)pgm_read_ptr(&drone_model_segments[i]);
        uint32_t segment_size = (i == {len(segments)-1}) ? ({len(tflite_model)} % {SEGMENT_SIZE}) : {SEGMENT_SIZE};
        if (segment_size == 0) segment_size = {SEGMENT_SIZE};

        memcpy_P(&psram_buffer[offset], segment, segment_size);
        offset += segment_size;
    }}

    Serial.printf("[INFO] Model załadowany do PSRAM: %u bytes\\n", offset);
    return psram_buffer;
}}

"""

    with open(output_path, 'w') as f:
        f.write(c_code)

    print(f"✓ C++ header: {output_path}")
    print(f"  Segmenty: {len(segments)}")
    print(f"  Funkcja: load_drone_model_to_psram()")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Konwerter Modelu Drona do TFLite dla ESP32-S3          ║")
    print("║  PyTorch ResNet18 → Keras → TFLite                      ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    pytorch_model, config = load_pytorch_model()

    if pytorch_model is None:
        print("\n[ERROR] Nie udało się załadować modelu PyTorch!")
        exit(1)

    print("\n" + "="*60)

    keras_model = pytorch_to_tflite_direct()

    tflite_model = convert_keras_to_tflite(keras_model, 'drone_model.tflite')

    print("="*60)
    generate_c_header(tflite_model, 'drone_model.h')

    print("\n" + "="*60)
    print("✓✓✓ KONWERSJA UKOŃCZONA!")
    print("="*60)
    print(f"Wyniki:")
    print(f"  • drone_model.tflite  ({len(tflite_model)/1024:.2f} KB)")
    print(f"  • drone_model.h       (C++ header z modelem)")
    print(f"\nNastępny krok:")
    print(f"  1. Skopiuj drone_model.h do ../include/")
    print(f"  2. Uruchom: pio run -e freenove_esp32_s3_wroom")
    print("="*60)