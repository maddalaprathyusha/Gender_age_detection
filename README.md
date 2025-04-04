Age and Gender Detection using OpenCV and Deep Learning

This project performs **real-time age and gender prediction** from images or webcam input using pre-trained deep learning models in OpenCV.

## 🧠 Project Overview

The goal is to detect human faces and classify:
- **Gender**: Male / Female
- **Age**: One of the 8 categories —  
  `(0-2)`, `(4-6)`, `(8-12)`, `(15-20)`, `(25-32)`, `(38-43)`, `(48-53)`, `(60-100)`
 Files in the Repository

| File Name                         | Description |
|----------------------------------|-------------|
| `age_gender_detection.py`        | Main Python script for real-time detection |
| `opencv_face_detector.pbtxt`     | Face detector configuration |
| `opencv_face_detector_uint8.pb`  | Face detector model |
| `age_deploy.prototxt`            | Age classification model architecture |
| `age_net.caffemodel`             | Pre-trained age classification model |
| `gender_deploy.prototxt`         | Gender classification model architecture |
| `gender_net.caffemodel`          | Pre-trained gender classification model |
| `README.md`                      | Project documentation ( which you’re reading it!) |

**Face Detection & Pre-trained Models**
Face detection is done using a DNN-based face detector from OpenCV (opencv_face_detector_uint8.pb and opencv_face_detector.pbtxt).

**CNN Architecture Used**
The architecture includes:

Convolutional Layer 1: 96 filters, kernel size 7×7

Convolutional Layer 2: 256 filters, kernel size 5×5

Convolutional Layer 3: 384 filters, kernel size 3×3

Fully Connected Layers: Two layers with 512 nodes each

Output Layer: Softmax layer for classification



**Age and gender prediction uses:**

.prototxt files → define the model architecture

.caffemodel files → contain the trained weights

**Models:**
Age: age_deploy.prototxt and age_net.caffemodel

Gender: gender_deploy.prototxt and gender_net.caffemodel

**Process of working:**

Detect faces in the input image or webcam feed.

Crop and preprocess the face region.

Pass it through the pre-trained CNN models.

Predict gender and age group.

Display results (bounding box + labels) on screen.


**Requirements**
Python 3.x

OpenCV (opencv-python)

NumPy

**Output:**

Mean values for model normalization: [78.4263377603, 87.7689143744, 114.895847746]

Classification is performed using Softmax.

The code supports image input and live webcam detection.

The model is classification-based, not regression, due to difficulty in exact age estimation.
