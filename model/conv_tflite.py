import tensorflow as tf

# Ask for .keras file path
model_path = input("Enter path to .keras model: ").strip()

# Load model
model = tf.keras.models.load_model(model_path)

# Ask for .tflite output path
tflite_path = input("Enter desired .tflite output filename: ").strip()
if not tflite_path.endswith(".tflite"):
    tflite_path += ".tflite"

# Convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved as: {tflite_path}")

