import tensorflow as tf
from tensorflow.python.keras import layers, models
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.engine import data_adapter

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



print(X_train.shape)

# Define CNN model
cnn = models.Sequential([
    # Images analysed will be 64x64x1 images, analysing the luminance.
    layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(64, 64, 1)), 
    layers.MaxPooling2D(pool_size=(2,2)),
    
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Prevent overfitting
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
cnn.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print model summary
cnn.summary()

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

# Train model
cnn.fit(
    X_train, 
    y_train, 
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)

y_pred = cnn.predict(X_val)
y_pred[:5]


print(y_pred[:10])
print(y_val[:10])


'''
from keras import __version__
tf.keras.__version__ = __version__
print(tf.keras.__version__)
# Save trained model
cnn.save("fish_detector.h5")


# Convert model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(cnn)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable quantization for speed
tflite_model = converter.convert()

# Save the TFLite model
with open("fish_detector.tflite", "wb") as f:
    f.write(tflite_model)
'''