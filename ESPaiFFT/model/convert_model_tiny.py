"""
Konwersja lekkiego modelu audio (DS-CNN) do TFLite INT8 dla ESP32-S3.

Wejście: 96x96x1 (Mel spektrogram zgodny z MelSpectrogramGenerator)
Wyjście: 2 klasy (drone / other)

Uwaga: to szkic architektury bez treningu. Aby uzyskać sensowną dokładność,
przeprowadź trening w Keras i wczytaj wytrenowane wagi przed konwersją.
"""

import os
import numpy as np
import tensorflow as tf

INPUT_SHAPE = (96, 96, 1)
NUM_CLASSES = 2


def depthwise_separable(x, filters, stride=1):
    x = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def build_tiny_dscnn():
    inputs = tf.keras.Input(shape=INPUT_SHAPE, dtype=tf.float32, name='spectrogram')
    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = depthwise_separable(x, 24, stride=2)
    x = depthwise_separable(x, 32, stride=2)
    x = depthwise_separable(x, 48, stride=2)
    x = depthwise_separable(x, 64, stride=2)
    x = tf.keras.layers.AveragePooling2D(pool_size=3)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def representative_dataset():
    for _ in range(200):
        spectrogram = np.random.rand(1, *INPUT_SHAPE).astype(np.float32)
        yield [spectrogram]


def convert_to_tflite(model, out_path='drone_model.tflite'):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Zapisano: {out_path} ({len(tflite_model)/1024:.1f} KB)")
    return tflite_model


def generate_header(tflite_model: bytes, out_path='drone_model.h'):
    SEG = 1024
    segments = []
    for i in range(0, len(tflite_model), SEG):
        chunk = tflite_model[i:i+SEG]
        hex_str = ', '.join(f'0x{b:02x}' for b in chunk)
        segments.append(hex_str)

    c = f"""// Auto-generated tiny TFLite model header
// Size: {len(tflite_model)} bytes ({len(tflite_model)/(1024*1024):.2f} MB)
// Input: [1, {INPUT_SHAPE[0]}, {INPUT_SHAPE[1]}, 1] INT8
// Output: [1, {NUM_CLASSES}] INT8




"""
    for i, seg in enumerate(segments):
        c += f"""
const uint8_t drone_model_segment_{i}[] PROGMEM = {{
    {seg}
}};
"""
    c += f"""
const uint8_t* const drone_model_segments[DRONE_MODEL_SEGMENTS] PROGMEM = {{
"""
    for i in range(len(segments)):
        c += f"    drone_model_segment_{i},\n"
    c += f"""}};

uint8_t* load_drone_model_to_psram() {{
    uint8_t* buf = (uint8_t*) ps_malloc(DRONE_MODEL_TOTAL_SIZE);
    if (!buf) {{
        Serial.println("[ERROR] PSRAM alloc failed for model");
        return nullptr;
    }}
    uint32_t off = 0;
    for (int i = 0; i < DRONE_MODEL_SEGMENTS; i++) {{
        const uint8_t* seg = (const uint8_t*) pgm_read_ptr(&drone_model_segments[i]);
        uint32_t sz = (i == DRONE_MODEL_SEGMENTS - 1) ? (DRONE_MODEL_TOTAL_SIZE % DRONE_MODEL_SEGMENT_SIZE) : DRONE_MODEL_SEGMENT_SIZE;
        if (sz == 0) sz = DRONE_MODEL_SEGMENT_SIZE;
        memcpy_P(&buf[off], seg, sz);
        off += sz;
    }}
    Serial.printf("[INFO] Model loaded to PSRAM: %u bytes\n", off);
    return buf;
}}

"""
    with open(out_path, 'w') as f:
        f.write(c)
    print(f"Zapisano: {out_path}")


def main():
    print("Tworzę lekki model DS-CNN 96x96x1...")
    model = build_tiny_dscnn()
    print("Konwertuję do TFLite INT8...")
    tfl = convert_to_tflite(model, 'drone_model.tflite')
    print("Generuję nagłówek C++...")
    generate_header(tfl, 'drone_model.h')
    print("Gotowe. Skopiuj drone_model.h do ../include i ustaw USE_DRONE_MODEL=1 w platformio.ini")


if __name__ == '__main__':
    main()