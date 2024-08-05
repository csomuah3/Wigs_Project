Face Shape Classification and Wig Recommendation System
youtube video:https://youtu.be/CrOTY5pD8w0
Overview

This project aims to classify face shapes from images and recommend suitable wigs based on the detected face shape. It consists of multiple scripts for different functionalities including image processing, model training, and web application for user interaction.

Project Structure

- `app.py`: Flask web application for face shape classification and wig recommendation.
- `create.py`: Script to generate a labeled dataset of face images.
- `face_shape.py`: Script for feature extraction, model training, and face shape prediction.
- `image_con.py`: Script to convert images in the dataset to JPEG format.
- `wigs_rename.py`: Script to rename wig images based on the detected face shape.
- `face_shape_features.csv`: CSV file containing extracted features from face images for model training.

Setup and Installation

Clone the repository:
git clone <repository-url>
cd <repository-directory>

Install the required dependencies:

pip install -r requirements.txt

Download necessary models:

- Download `shape_predictor_68_face_landmarks.dat` from [dlib model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the `models` directory.
- Place the trained model `face_shape_model.pkl` in the `models` directory.

Scripts and Usage

Flask Web Application (`app.py`)

This script initializes a Flask web application to classify face shapes from user-uploaded images and recommend wigs based on the detected face shape.

python app.py

Endpoints:

- `/`: Main page to upload images and display results.
- `/uploads/<filename>`: Serve uploaded images for preview.

Create Dataset (`create.py`)

This script processes a directory of face images, renames them based on their labels, and creates a CSV file (`labels.csv`) containing image names and their corresponding labels.


python create.py

Face Shape Classification (`face_shape.py`)

This script extracts features from face images, trains a Random Forest classifier, and evaluates the model. It also includes functionality to predict face shapes from images.


python face_shape.py

Functions:

- `extract_features(image_path)`: Extracts landmark distances from an image.
- `create_dataset(image_dir, labels_file)`: Creates a dataset from labeled images.
- `predict_face_shape(image_path)`: Predicts the face shape from an image.

Image Conversion (`image_con.py`)

This script converts all images in a specified folder to JPEG format.

python image_con.py
Wig Image Renaming (`wigs_rename.py`)

This script renames wig images based on the detected face shape.

bash
python wigs_rename.py

Directory Structure

- `models/`: Directory containing the trained model and shape predictor.
- `static/uploads/`: Directory for storing uploaded images.
- `static/wigs_images/`: Directory containing wig images.

Dependencies

- Python 3.x
- Flask
- OpenCV
- dlib
- scikit-learn
- numpy
- pandas
- matplotlib
- Pillow

Example Usage

Run the web application

bash
python app.py

Upload an image through the web interface. The application will classify the face shape and recommend suitable wigs.


