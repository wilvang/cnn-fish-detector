import cv2
import numpy as np
import datetime
import tflite_runtime.interpreter as tflite
import os

# Oppgi filnavnene til modell og logg
MODEL_PATH = "fish_detector.tflite"
LOG_FILE = "detections.log"

# Initiell konfigurasjon
IMG_HEIGHT = 64
IMG_WIDTH = 64

def preprocess_frame(frame):
    """
    Gjør samme forprosessering som i treningskoden:
      1. Konverter BGR -> YCrCb
      2. Ta kun Y-kanalen (index 0)
      3. Resizer til (64,64)
      4. Normaliserer til [0,1]
      5. Returnerer et numpy-array med shape (1,64,64,1)
    """
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel = ycbcr[:, :, 0]
    resized = cv2.resize(y_channel, (IMG_WIDTH, IMG_HEIGHT))
    normalized = resized.astype(np.float32) / 255.0

    # Legg til batch-dimensjon og 'channel' dim: (1,64,64,1)
    input_data = np.expand_dims(normalized, axis=(0, -1))  
    return input_data

def load_tflite_model(model_path):
    """
    Laster TFLite-modellen og returnerer en tflite.Interpreter.
    """
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def main():
    # Last TFLite-modellen
    interpreter = load_tflite_model(MODEL_PATH)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Åpne kamera (ID 0), juster evt. CAP_V4L2 om nødvendig
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kunne ikke åpne kamera. Avslutter.")
        return

    # Åpne loggfil i "append"-modus
    log_f = open(LOG_FILE, "a")

    try:
        print("Trykk Ctrl+C eller lukk vinduet for å avslutte.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Kunne ikke lese fra kamera.")
                break

            # Forprosessér ramme
            input_data = preprocess_frame(frame)

            # Kjør inferens med TFLite
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            # Siden modellen har 'Dense(1, sigmoid)', vil output være en verdi mellom [0,1].
            #  > 0.5 = "fish", ellers = "no_fish".
            score = prediction[0][0]
            if score >= 0.5:
                label = "fish"
            else:
                label = "no_fish"

            # Skriv label på bildet (valgfritt for visning)
            cv2.putText(
                frame,
                f"Prediction: {label} ({score:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            # Vis resultatet i et vindu (valgfritt)
            cv2.imshow("Fish Detector", frame)
            
            # Logg resultatet (med tidsstempel)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"{timestamp}, {label}, score={score:.4f}\n"
            log_f.write(log_line)
            log_f.flush()  # Sikre at data blir skrevet til fil

            # Avslutt hvis 'q' trykkes i vinduet
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nAvslutter ved Ctrl+C...")

    finally:
        # Lukk ressurser
        cap.release()
        cv2.destroyAllWindows()
        log_f.close()

if __name__ == "__main__":
    main()
