import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(X_train, y_train, X_val, y_val):
    # Define model
    cnn = models.Sequential([
        layers.Input(shape=(64, 64, 1)),
        layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'), 
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

    #data_adapter._is_distributed_dataset = _is_distributed_dataset

    # Train model
    cnn.fit(
        X_train, 
        y_train, 
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val)
    )

    return cnn
