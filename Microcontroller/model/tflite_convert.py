
import tensorflow as tf
import numpy as np

def representative_dataset():
    for _ in range(10):
        yield [np.random.randn(1, 1, 224, 224).astype(np.float32)]

class SimpleModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(512)
        self.dense2 = tf.keras.layers.Dense(2)

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 1, 224, 224], dtype=tf.float32)])
    def __call__(self, x):
        x = tf.reshape(x, [1, -1])
        x = self.dense1(x)
        x = tf.nn.relu(x)
        x = self.dense2(x)
        return x

model = SimpleModel()
concrete_func = model.__call__.get_concrete_function()

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open('Microcontroller/model/drone_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"✓ Generated TFLite model: {len(tflite_model) / 1024:.2f} KB")