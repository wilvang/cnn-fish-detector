import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from preprocess import augment_images
from segment import load_images
from train import save_model, build_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def main():
    dataset_path = "../data/"
    categories = ["fish", "no_fish"]

    X, y = load_images(dataset_path, categories)
    print(X.shape)
    # Split into training, validation, and testing sets
    X_model, X_test, y_model, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_model, y_model, test_size=0.2, random_state=42)

    augment_images(X_train, y_train)

    cnn = build_model(X_train, y_train, X_val, y_val)

    y_pred = cnn.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

    # Generate a classification report
    report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
    print(report)

if __name__ == "__main__":
    main()