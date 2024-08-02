import os
import joblib
import numpy as np
import cv2
import dlib

# Load the trained model
model = joblib.load('./models/face_shape_model.pkl')

# Initialize face detector and shape predictor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

# Function to extract features
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    features = []
    for (x, y, w, h) in faces:
        face_rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = predictor(gray, face_rect)
        distances = []
        for i in range(68):
            for j in range(i + 1, 68):
                dist = np.linalg.norm(
                    np.array([landmarks.part(i).x, landmarks.part(i).y]) - np.array([landmarks.part(j).x, landmarks.part(j).y]))
                distances.append(dist)
        features.append(distances)

    return features

# Function to classify face shape
def classify_face_shape(image):
    features = extract_features(image)
    if not features:
        return None
    features = np.array(features)
    prediction = model.predict(features)
    return prediction[0]

# Path to the folder containing wig images
wig_images_folder = 'wigs_images'

# Dictionary to keep track of counts for each face shape
face_shape_counts = {}

# Iterate through all images in the folder
for filename in os.listdir(wig_images_folder):
    if filename.lower().endswith(('jpg', 'jpeg', 'png')):
        file_path = os.path.join(wig_images_folder, filename)
        image = cv2.imread(file_path)
        
        # Check if the image is successfully loaded
        if image is None:
            print(f"Failed to load {filename}")
            continue
        
        # Classify face shape
        face_shape = classify_face_shape(image)
        if face_shape:
            # Initialize count for the face shape if not present
            if face_shape not in face_shape_counts:
                face_shape_counts[face_shape] = 1
            else:
                face_shape_counts[face_shape] += 1

            # Generate new filename with face shape and count
            base, ext = os.path.splitext(filename)
            new_filename = f"{face_shape.lower()}_{face_shape_counts[face_shape]}{ext}"
            new_file_path = os.path.join(wig_images_folder, new_filename)

            # Rename the image file
            try:
                os.rename(file_path, new_file_path)
                print(f"Renamed {filename} to {new_filename}")
            except Exception as e:
                print(f"Failed to rename {filename}: {e}")
        else:
            print(f"No face detected in {filename}")
