# Content-Based Image Retrieval Using CNN and Hash

## Introduction

This project implements a Content-Based Image Retrieval (CBIR) system that allows users to upload an image and find similar images from the Caltech-101 dataset. The similarity between images is determined by extracting features using a Convolutional Neural Network (CNN) and comparing these features using hash.

## Implementation

The project uses a pre-trained MobileNetV2 model to extract features from images. The extracted features are then converted into binary codes using a hash function, and the similarity between images is computed by comparing these binary codes.

## Setup Instructions

### 1. Create a Virtual Environment

It is recommended to use a virtual environment to manage the project's dependencies. Create a virtual environment using the following command:

```sh
python -m venv venv
```

### 2. Activate the Virtual Environment

Activate the virtual environment using the following command:

```sh
venv\Scripts\activate
```

### 3. Install Required Libraries

Install the necessary libraries using the requirements.txt file:

```sh
pip install -r requirements.txt
```
or

```sh
pip install flask numpy pandas matplotlib pillow keras tensorflow
```

### 4. Train the Model and Generate Binary Codes

Before running the application, you need to train the model and generate the binary codes for the image dataset. Execute the train.py file with the following command:

```sh
python train.py
```

This will train the feature extraction model and create t binary codes for the images.

### 5. Run the Flask Application

After generating the binary codes, you can run the Flask application. Execute the app.py file with the following command:

```sh
python app.py
```
### 6. Access the Application

Open your web browser and go to the following URL to access the CBIR application:

```sh
http://127.0.0.1:5000/
```
## Usage

1. **Upload an Image:** On the main page, upload an image by clicking the "Choose File" button and selecting an image from your computer.

2. **View Results:** After uploading, the application will process the image and display the query image along with the top 10 most similar images from the dataset.

3. **Search Another Image:** You can upload another image by clicking the "Upload Another Image" button on the results page.

## Conclusion

This project demonstrates how to build a CBIR system using Flask and a pre-trained CNN model for feature extraction. The similarity search is performed by comparing binary codes of image features, offering an efficient way to find similar images in a dataset.
