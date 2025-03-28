import preprocess
import segment
import train
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import tensorflow as tf

def main():
    dataset_path = "../data/"
    categories = ["fish", "no_fish"]

    X, y = segment.load_images(dataset_path, categories)
   
    # Split into training, validation, and testing sets
    X_model, X_test, y_model, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_model, y_model, test_size=0.2, random_state=42)
    
    # preprocess.augment_images(X_train, y_train)
    
    cnn = train.build_model(X_train, y_train, X_val, y_val)
    cnn.evaluate(X_test, y_test)

    y_pred = cnn.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
    y_pred = y_pred.ravel()
    # Generate a classification report
    report = classification_report(y_test, y_pred, target_names=['fish', 'no_fish'])
    print(report)

    plt.imshow(X_test[2])
    plt.xlabel(categories[y_test[2]] + " or " + categories[y_pred[2]])


    cnn.save("fish_detector.keras", include_optimizer=True)

if __name__ == "__main__":
    main()