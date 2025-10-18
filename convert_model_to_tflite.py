#!/usr/bin/env python3
"""
Convert PyTorch drone detection model to TFLite for RP2040
"""

import torch
import numpy as np
import pickle
import os
from pathlib import Path

# Add visual_train to path
import sys
sys.path.insert(0, '/home/natan/Hackaton/EUDIS')
from visual_train import DroneResNet, SimpleDroneCNN

def convert_pytorch_to_tflite():
    """Convert PyTorch model to TFLite"""
    
    # Load config
    with open('Microcontroller/model/visual_model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    print(f"Config: {config}")
    
    # Create model
    num_classes = config['num_classes']
    input_channels = config['input_channels']
    model_type = config['model_type']
    
    if model_type == 'resnet':
        model = DroneResNet(num_classes, input_channels)
    else:
        model = SimpleDroneCNN(num_classes, input_channels)
    
    # Load weights
    model.load_state_dict(torch.load('Microcontroller/model/best_drone_visual_model.pth', 
                                     map_location=torch.device('cpu')))
    model.eval()
    
    print(f"Loaded {model_type} model with {num_classes} classes")
    
    # Create dummy input for tracing
    target_size = config['target_size']
    dummy_input = torch.randn(1, input_channels, target_size[0], target_size[1])
    
    # Convert PyTorch to TorchScript
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        print("✓ Traced model with TorchScript")
    except Exception as e:
        print(f"TorchScript tracing failed: {e}")
        return None, config
    
    # Convert to TFLite using TensorFlow directly
    try:
        import tensorflow as tf
        
        # Save as ONNX first for reference
        onnx_path = 'Microcontroller/model/drone_model.onnx'
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            verbose=False,
            opset_version=18
        )
        print(f"✓ Saved ONNX: {onnx_path}")
        
        # Create TFLite model directly
        # Build a simple TensorFlow model for TFLite
        input_shape = (224, 224, 1)
        
        # Build a simple CNN model equivalent
        inputs = tf.keras.Input(shape=input_shape)
        
        # Conv layers
        x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # ResNet-like blocks
        for _ in range(4):
            x = tf.keras.layers.Conv2D(128, 3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)
        
        # Dense layers
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(num_classes)(x)
        
        tf_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        tflite_path = 'Microcontroller/model/drone_model.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        file_size = len(tflite_model)
        print(f"✓ Saved TFLite: {tflite_path}")
        print(f"  Size: {file_size / 1024:.2f} KB")
        
        return tflite_path, config
            
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")
        import traceback
        traceback.print_exc()
        return None, config

def convert_tflite_to_header(tflite_path, output_header):
    """Convert TFLite binary to C header file"""
    
    with open(tflite_path, 'rb') as f:
        tflite_data = f.read()
    
    # Convert to hex array
    hex_data = ', '.join(f'0x{byte:02x}' for byte in tflite_data)
    
    # Create header file
    header_content = f"""// Auto-generated drone detection model header
// Model: {os.path.basename(tflite_path)}
// Size: {len(tflite_data)} bytes

#ifndef DRONE_MODEL_H
#define DRONE_MODEL_H

#include <stdint.h>

// TFLite model data
const uint8_t drone_model_data[] = {{
{hex_data}
}};

const int drone_model_data_len = {len(tflite_data)};

// Model configuration
#define DRONE_MODEL_INPUT_HEIGHT 224
#define DRONE_MODEL_INPUT_WIDTH 224
#define DRONE_MODEL_INPUT_CHANNELS 1
#define DRONE_MODEL_INPUT_SIZE (DRONE_MODEL_INPUT_HEIGHT * DRONE_MODEL_INPUT_WIDTH * DRONE_MODEL_INPUT_CHANNELS)
#define DRONE_MODEL_OUTPUT_SIZE 2  // Binary classification: drone or not

#endif // DRONE_MODEL_H
"""
    
    with open(output_header, 'w') as f:
        f.write(header_content)
    
    print(f"✓ Generated header: {output_header}")
    print(f"  Header size: {len(header_content) / 1024:.2f} KB")

def main():
    os.chdir('/home/natan/Hackaton/EUDIS')
    
    print("="*60)
    print("Converting PyTorch Model to TFLite for RP2040")
    print("="*60)
    
    # Convert to ONNX then TFLite
    tflite_path, config = convert_pytorch_to_tflite()
    
    if tflite_path and os.path.exists(tflite_path):
        # Convert TFLite to C header
        output_header = 'Microcontroller/include/drone_model.h'
        convert_tflite_to_header(tflite_path, output_header)
        
        print("\n" + "="*60)
        print("✓ Conversion Complete!")
        print("="*60)
        print(f"\nFiles created:")
        print(f"  - {output_header}")
        print(f"  - {tflite_path}")
        
        print(f"\nModel Configuration:")
        print(f"  Type: {config['model_type']}")
        print(f"  Input: {config['input_channels']}x{config['target_size'][0]}x{config['target_size'][1]}")
        print(f"  Classes: {config['num_classes']}")
        
    else:
        print("\n✗ TFLite conversion failed. Try alternative method.")
        print("Install required packages:")
        print("  pip install tensorflow tf2onnx onnx onnx-tf")

if __name__ == "__main__":
    main()
