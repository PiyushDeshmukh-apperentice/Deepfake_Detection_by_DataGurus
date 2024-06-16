
# Deepfake Detection using CNN and Streamlit

  

## Project Overview

  

This project aims to detect deepfake videos using a Convolutional Neural Network (CNN) model integrated with a Streamlit web application. The model analyzes frames from a video, detects faces, and classifies them as either real or deepfake. The computational power of Google Colab is leveraged for model inference, while a user-friendly interface is provided through Streamlit.

  

## Requirements

  

- Python 3.7+

- Numpy

- Scikit

- TensorFlow 2.x

- OpenCV

- MTCNN (Multi-Task Cascaded Convolutional Neural Networks)

- Flask

- Streamlit

- ngrok

  

## Setup Instructions

  
1. **Download Dataset from Kaggle**

- Refer the following dataset for using the model "[DataSet for Deepfake](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images/data)"


2. **Set Up Google Colab**

  

- Open the provided Jupyter Notebook `DeepfakeDetection_Finale.ipynb` in Google Colab.

- Ensure that the notebook is connected to a GPU T4 runtime.

- Install the necessary libraries within the Colab environment.

- Run the notebook to load the model and start the Flask server.

  



## Running the Project

  

1. **Run the Colab Notebook**

  

- Execute all cells in the `Deepfake_Detection_byDataGurus_ResultAnalysis.ipynb` notebook to start the Flask server.

  
## Using the Streamlit Application

  

1. **Upload Video**

  

- Use the file uploader in the Streamlit app to upload a video file in mp4, mov, avi, or mkv format.

  

2. **View Results**

  

- The app will process the video, detect faces, and classify the video as either real or deepfake. The result will be displayed on the webpage.

  

## Explanation of the Code

  

### Jupyter Notebook (`Deepfake_Detection_byDataGurus_ResultAnalysis.ipynb`)

  

- **Model Loading and Preprocessing**: The notebook loads the pre-trained CNN model and defines functions for image preprocessing and prediction.

- **Flask Server**: A Flask server is set up to expose an endpoint for predicting deepfake videos.

  

### Streamlit Application (`deepfakeDetection.py`)

  

- **File Upload**: Users can upload video files through the Streamlit interface.

- **Video Processing**: The uploaded video is sent to the Colab environment for processing using the Flask API.

- **Model on which Streamlit application is based**: `mDeepfake_Detection_byDataGurus_ResultAnalysis.ipynb` save this model with model.save("Name_of_model.h5") and use to run streamlit application


- **Result Display**: The prediction result is retrieved from the Colab environment and displayed to the user.


## Notes and References :
- Please refer Docs.md for further technicalities and function understandings of each file.
  

## Contributing

  

We welcome contributions! 

For Contributions you can visit the LinkedIn accounts of the creators of this file as given below : 

[Prasanna Patwardhan](https://www.linkedin.com/in/prasanna-patwardhan-24782228b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) 

[Piyush Deshmukh](https://www.linkedin.com/in/piyush-deshmukh145?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)

[Yash Kulkarni](https://www.linkedin.com/in/yash-kulkarni-7996neno?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)

[Yugandhar Chawale](https://www.linkedin.com/in/yugandharchawale?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)

