import os
import cv2
import numpy as np
from multiprocessing import Pool

def process_image(args):
    img_path, label = args
    img = cv2.imread(img_path)
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img_y = img_ycbcr[:, :, 0]
    img_resized = cv2.resize(img_y, (64, 64))
    return img_resized, label

def load_images(dataset_path, categories):
    X, y = [], []
    args_list = []
    for label, category in enumerate(categories):
        folder_path = os.path.join(dataset_path, category)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            args_list.append((img_path, label))
    
    with Pool() as pool:
        results = pool.map(process_image, args_list)
    
    for img_resized, label in results:
        X.append(img_resized)
        y.append(label)
    
    X = np.array(X).reshape(-1, 64, 64, 1)  # Reshape for CNN input (1 channel)
    X = X / 255.0  # Normalize pixel values to [0,1]
    y = np.array(y)
    
    return X, y