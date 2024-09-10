
# Hand Gesture Recognition / Sign Language Recognition System

This repository contains the implementation of a Hand Gesture Recognition System using computer vision techniques and machine learning. The system aims to recognize and interpret hand gestures in real-time, making it ideal for applications such as human-computer interaction, sign language recognition, and gesture-based control systems.


## Features
- Real-Time Gesture Recognition: Detects and recognizes hand gestures from live video input.
- Multi-Gesture Support: Supports a variety of hand gestures for different use cases.
- OpenCV & Machine Learning: Utilizes OpenCV for image processing and machine learning algorithms for gesture classification.
- Customizable: Easily extendable to add new gestures or improve the accuracy of the model.

## How It Works
- Hand Detection: Captures frames from the camera and detects the hand region.
- Feature Extraction: Processes the detected hand region to extract key features such as contours and finger positions.
- Gesture Classification: Classifies the extracted features into predefined gestures using a trained machine learning model.
- Gesture Output: Displays the recognized gesture and can trigger corresponding actions.


## Requirements
- Python 3.x
- OpenCV
- Numpy
- TensorFlow/Keras (or other machine learning libraries)
- MediaPipe
## Gesture Names
- okay
- peace
- thumbs up
- thumbs down
- call me
- stop
- rock
- live long
- fist
- smile
