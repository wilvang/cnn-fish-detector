import tensorflow as tf
from tensorflow.python.keras import layers, models
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# Set dataset path
dataset_path = "dataset/"
categories = ["no_fish", "fish"]

# Load images and labels
X, y = [], []

for label, category in enumerate(categories):
    folder_path = os.path.join(dataset_path, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        
        # Read image and convert to YCbCr
        img = cv2.imread(img_path)
        img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        
        # Use only the Y (luminance) channel
        img_y = img_ycbcr[:, :, 0]
        
        # Resize to match CNN input
        img_resized = cv2.resize(img_y, (64, 64))  # Adjust size as needed

        X.append(img_resized)
        y.append(label)

# Convert to NumPy arrays
X = np.array(X).reshape(-1, 64, 64, 1)  # Reshape for CNN input (1 channel)
y = np.array(y)

# Normalize pixel values to [0,1]
X = X / 255.0

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model
model = models.Sequential([
    # Images analysed will be 64x64x1 images, analysing the luminance.
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(64, 64, 1)), 
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Prevent overfitting
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()


# Train model
history = model.fit(
    X_train, y_train,
    epochs=15, batch_size=16,
    validation_data=(X_val, y_val)
)

# Save trained model
model.save("fish_detector.h5")

# Convert model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable quantization for speed
tflite_model = converter.convert()

# Save the TFLite model
with open("fish_detector.tflite", "wb") as f:
    f.write(tflite_model)


