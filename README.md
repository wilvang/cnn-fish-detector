# Low-Cost Fish Detection in Live Video

## Overview
This project aims to develop a low-cost fish detection model that can run on a Raspberry Pi. The model will utilize live video surveillance of rivers to detect the presence of fish, automating the monitoring of fish populations in Norwegian rivers. The project is a collaboration with Anadrom and Nordavind Utvikling AS, who are providing labeled training data captured from monitoring stations deployed nationwide.

## Objectives
- Detect fish (i.e., fish / no fish) in video.
- Implement a technique that can detect fish in a video and run on low-cost equipment.
- Process each frame of the video input to the Raspberry Pi and output fish/no fish.
- Assess the impact of parameters such as resolution, frame rate, and visibility conditions.

## Data Collection
Training data is provided by Anadrom and Nordavind Utvikling AS. The data consists of labeled video frames captured from monitoring stations across Norwegian rivers.

## Methodology
1. **Image Preprocessing**: 
   - Expand the training data using techniques such as data augmentation.
   - Reduce noise in deployment using filters and other preprocessing techniques.

2. **Model Training**:
   - Train an AI model using the preprocessed data.
   - Evaluate the model's performance and fine-tune hyperparameters.

3. **Deployment**:
   - Implement the trained model on a Raspberry Pi.
   - Process live video input and output fish/no fish detection results.

## Requirements
- Raspberry Pi
- Python 3.11
- OpenCV
- TensorFlow/Keras
- Labeled training data from Anadrom and Nordavind Utvikling AS
