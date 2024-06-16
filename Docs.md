
# Deepfake Detection Project Documentation

  

  

## Overview

  

  

This document provides detailed information about the functions, classes, and important code snippets used in the Deepfake Detection project.

  

  

## Functions and Classes

  

  

### 1. `load_images_from_folder(folder, max_images=100)`

  

  

**Description**: Loads images from a specified folder and resizes them.

  

  

**Parameters**:

  

-  `folder` (str): Path to the folder containing images.

  

-  `max_images` (int): Maximum number of images to load from the folder.

  

  

**Returns**:

  

-  `images` (list): List of resized images.

  

  

### 2. `load_dataset(dataset_path, max_images_per_class=100)`

  

  

**Description**: Loads and combines real and fake images into a dataset.

  

  

**Parameters**:

  

-  `dataset_path` (str): Path to the dataset directory containing 'Real' and 'Fake' subdirectories.

  

-  `max_images_per_class` (int): Maximum number of images to load per class.

  

  

**Returns**:

  

-  `images` (numpy.ndarray): Array of images.

  

-  `labels` (numpy.ndarray): Array of corresponding labels (0 for real, 1 for fake).

  

  

#####

  

  

### 3. `preprocess_images(images)`

  

  

**Description**: Normalizes pixel values of images to the range [0, 1].

  

  

**Parameters**:

  

-  `images` (numpy.ndarray): Array of images to preprocess.

  

  

**Returns**:

  

-  `normalized_images` (numpy.ndarray): Array of normalized images.

  

  

### 4. `build_model(input_shape)`

  

  

**Description**: Builds a Convolutional Neural Network (CNN) model.

  

  

**Parameters**:

  

-  `input_shape` (tuple): Shape of the input images.

  

  

**Returns**:

  

-  `model` (tf.keras.Sequential): Compiled CNN model.

  

  

### Explanation of Keras Components

  

  

#### Keras Layers and Functions

  

  

#### 1. `Conv2D`

  

  

**Description**: `Conv2D` is a 2D convolutional layer that creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. It is commonly used in image processing tasks to detect features such as edges, textures, and shapes.

  

  

**Parameters**:

  

- filters (int): Number of output filters in the convolution.

  

- kernel_size (tuple): Size of the convolution kernel.

  

- activation (str): Activation function to use (e.g., 'relu').

  

- input_shape (tuple, optional): Shape of the input data (only specified for the first layer).

  

  

#### 2. MaxPooling2D

  

  

**Description**: MaxPooling2D is a pooling layer that performs max pooling operation for spatial data. It reduces the spatial dimensions (width and height) of the input volume, helping to reduce the number of parameters and computation in the network.

  

  

**Parameters**:

  

  

- pool_size (tuple): Size of the pooling window.

  

- strides (tuple, optional): Strides of the pooling operation.

  

  

#### 3. Flatten

  

  

**Description**: Flatten is a layer that flattens the input, i.e., converts the input into a 1D array. It is often used to transition from the convolutional layers to the fully connected layers in a CNN.

  

  

#### 4. Dense

  

  

**Description**:Dense is a fully connected layer that connects every neuron in the previous layer to every neuron in the current layer. It is used to perform high-level reasoning in the network.

  

  

**Parameters**:

  

  

- units (int): Number of neurons in the layer.

  

- activation (str, optional): Activation function to use (e.g., 'relu', 'sigmoid').

  

  

### 5. `predict_image(image, model)`

  

  

**Description**: Predicts whether an image is real or a deepfake using the trained model.

  

  

**Parameters**:

  

-  `image` (numpy.ndarray): Image to predict.

  

-  `model` (tf.keras.Model): Trained CNN model.

  

  

**Returns**:

  

-  `result` (str): "Real" or "Deepfake".

  

  

### 6. `detect_faces_in_video(video_path, output_path, model)`

  

  

**Description**: Detects faces in a video using MTCNN, processes the frames, and predicts whether the video is real or a deepfake.

  

  

**Parameters**:

  

-  `video_path` (str): Path to the input video file.

  

-  `output_path` (str): Path to save the processed video with detected faces.

  

-  `model` (tf.keras.Model): Trained CNN model.

  

  

**Returns**:

  

-  `result` (str): "Real", "Deepfake", or "No faces detected in the video".

  

### 7. `predict_for_image(image_path, model)`

  

**Description**: This function predicts if an image is real or fake

  

**Parameters**:

  

-  `image_path` : The path to the image to be predicted.

  

-  `model` (tf.keras.Model): Trained CNN model.

  
  

**Returns**:

  

-  `result` (str): "Real", "Deepfake", or "No faces detected in the video".

  

### 7. `convert_to_jpg(image_path)`

  

**Description**: This function converts an image to JPEG format if it is not already in that format

  

**Parameters**:

  

-  `image_path` : The path to the image to be predicted.

  
  
  

**Returns**:

  

-  `result` Image: The image in JPEG format., so that it can be feeded into the function predict_for_image



## Streamlit Application functions and structure :

### 1. `is_deepfake(file_path, model)`

  

  

**Description**: The `is_deepfake` function processes video files to detect deepfakes using a pre-trained deep learning model. It scans each frame of the video for faces, analyzes these faces, and determines if any of them are deepfakes.
	- initializes an MTCNN detector to detect faces. using MTCNN()
	- read frames until the video ends.
	- put each frame which detects faces into the model to detect deepfakes


**Parameters**:
-   **`file_path`** (str): Path to the video file to be analyzed.
-   **`model`** (tf.keras.Model): Pre-trained TensorFlow deepfake detection model

**Returns**:
**`bool`**: Returns `True` if a deepfake is detected, otherwise `False`.


### 2. `st.file_uploader()`


**Description**: Allows users to upload files of specified types (JPEG, PNG, MP4, AVI). The uploaded file is read into memory as bytes.
	- If the uploaded file is an image (JPEG or PNG), the bytes are converted to a NumPy array and displayed with `st.image`. The image is processed using OpenCV and MTCNN to detect faces. 
	- For videos, each frame is processed until the video ends and each detected face is extracted, resized, and analyzed by the deepfake detection model.

