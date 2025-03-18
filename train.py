import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_dir, val_dir):
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary'
    )
    
    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(128, 128),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary'
    )
    
    model.fit(train_generator, epochs=10, validation_data=val_generator)

if __name__ == "__main__":
    train_dir = "data/processed/train"
    val_dir = "data/processed/val"
    model = build_model()
    train_model(model, train_dir, val_dir)
    model.save("fish_detection_model.h5")
