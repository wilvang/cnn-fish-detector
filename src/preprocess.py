import cv2
import numpy as np
import random


def gaussian_noise(img):
    return cv2.GaussianBlur(img, (5, 5), 0)


def rotate_image(img):
    h, w = img.shape[:2]
    angle = random.randint(0, 360)  # Random angle between 0 and 360 degrees
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h)) # Returns the rotated image


def mirror(img):
    return cv2.flip(img, 1)


def contrast(img):
    factor = random.randint(0, 200)
    return cv2.convertScaleAbs(img, alpha=(factor/100), beta=0)


def brightness(img):
    factor = random.randint(0, 200)
    img_255 = (img * 255).astype(np.uint8)
    
    # Apply brightness adjustment
    brightened = cv2.convertScaleAbs(img_255, alpha=1, beta=int((factor - 1) * 255))
    
    # Scale image values back to 0-1
    return brightened / 255.0


def zoom(img):
    zoom_factor = random.uniform(0.8, 1.2)
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
   

def augment_images(X, y):
    techniques = [gaussian_noise, rotate_image, mirror, brightness, zoom]
    X_augmented, y_augmented = [], []

    for img, label in zip(X, y):
        for technique in techniques:
            img_augmented = technique(img)
            # Ensure the augmented image has the same shape and type as the original
            img_augmented = cv2.resize(img_augmented, (64, 64))
            img_augmented = img_augmented.astype(np.float32)  # Ensure consistent type
            X_augmented.append(img_augmented)
            y_augmented.append(label)
    
    X_augmented = np.array(X_augmented).reshape(-1, 64, 64, 1)
    y_augmented = np.array(y_augmented)
    
    X_expanded = np.concatenate((X, X_augmented), axis=0)
    y_expanded = np.concatenate((y, y_augmented), axis=0)
    
    return X_expanded, y_expanded
