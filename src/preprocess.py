import cv2
import numpy as np
import os

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Resize the image
    resized_image = cv2.resize(blurred_image, (128, 128))
    
    return resized_image

def preprocess_data(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            processed_image = preprocess_image(image_path)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, processed_image)

if __name__ == "__main__":
    input_dir = "data/raw"
    output_dir = "data/processed"
    preprocess_data(input_dir, output_dir)
