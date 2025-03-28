import cv2
import tensorflow as tf
import numpy as np
import main


def detect_fish(frame, model):
    frame_ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    frame_y = frame_ycbcr[:, :, 0]
    resized_frame = cv2.resize(frame_y, (64, 64))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=(0, -1))
    
    prediction = model.predict(input_frame)
    return prediction[0][0] > 0.5

def capture():
    # model_path = "fish_detection_model.h5"
    # model = load_model(model_path)
    
    #model = main.main()

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if detect_fish(frame, model):
            print("Fish detected!")
        else:
            print("No fish detected.")
        
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture()