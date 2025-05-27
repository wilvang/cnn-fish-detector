import cv2
import numpy as np
import datetime
import tflite_runtime.interpreter as tflite

# Constants
MODEL_PATH = "../model/fish_detector-v2.tflite"
LOG_FILE = "../logs/detections.log"
IMG_HEIGHT = 64
IMG_WIDTH = 64


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess the input frame to match model training:
      1. Convert BGR to YCrCb color space
      2. Extract the Y channel (index 0)
      3. Resize to (64, 64)
      4. Normalize pixel values to [0,1]
      5. Add batch and channel dimensions → shape (1,64,64,1)
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0]
    resized = cv2.resize(y_channel, (IMG_WIDTH, IMG_HEIGHT))
    normalized = resized.astype(np.float32) / 255.0
    input_data = np.expand_dims(normalized, axis=(0, -1))
    return input_data


def load_tflite_model(model_path: str) -> tflite.Interpreter:
    """
    Load the TFLite model and allocate tensors.
    """
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_inference(interpreter: tflite.Interpreter, input_data: np.ndarray) -> float:
    """
    Run inference on input data and return the prediction score.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return float(prediction[0][0])


def main():
    # Load model
    interpreter = load_tflite_model(MODEL_PATH)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera. Exiting.")
        return

    # Open log file for appending
    with open(LOG_FILE, "a") as log_file:
        print("Press Ctrl+C or close the window to exit.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera.")
                    break

                # Preprocess frame
                input_data = preprocess_frame(frame)

                # Run inference
                score = run_inference(interpreter, input_data)

                # Interpret prediction
                label = "fish" if score >= 0.5 else "no_fish"

                # Display prediction on frame
                cv2.putText(
                    frame,
                    f"Prediction: {label} ({score:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

                # Show the frame
                cv2.imshow("Fish Detector", frame)

                # Log the result with timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"{timestamp}, {label}, score={score:.4f}\n")
                log_file.flush()

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nExiting on user interrupt...")

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
