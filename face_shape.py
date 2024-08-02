import dlib
import cv2
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
from collections import Counter
import matplotlib.pyplot as plt

import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module=".*")

# Initialize face detector and shape predictor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

def extract_features(image_path):
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        return [], False
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return [], False
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print(f"No faces detected in image: {image_path}")
        return [], False

    features = []
    for (x, y, w, h) in faces:
        face_rect = dlib.rectangle(x, y, x+w, y+h)
        landmarks = predictor(gray, face_rect)
        
        # Extract distances between landmarks
        distances = []
        for i in range(68):
            for j in range(i+1, 68):
                dist = np.linalg.norm(np.array([landmarks.part(i).x, landmarks.part(i).y]) - np.array([landmarks.part(j).x, landmarks.part(j).y]))
                distances.append(dist)
        
        features.append(distances)
    
    return features, True

def create_dataset(image_dir, labels_file):
    data = []
    labels = []
    
    df_labels = pd.read_csv(labels_file)
    label_dict = dict(zip(df_labels['image_name'], df_labels['label']))

    for image_name, label in label_dict.items():
        image_path = os.path.join(image_dir, image_name)
        print(f"Processing {image_path}")  # Debugging
        features, has_face = extract_features(image_path)
        if features:
            data.extend(features)
            labels.extend([label] * len(features))
        else:
            print(f"No features extracted for {image_path}")  # Debugging

    if not data:
        raise ValueError("No features extracted. Check your image directory and labels file.")
    
    # Debugging: Print the first few entries of data
    print("Sample of extracted features:", data[:5])
    
    # Check feature length
    feature_length = len(data[0])
    print(f"Feature length: {feature_length}")

    df = pd.DataFrame(data, columns=[f'dist_{i}' for i in range(feature_length)])
    df['label'] = labels
    return df

# Example usage
image_dir = 'new_training_set'
labels_file = 'labels.csv'

try:
    df = create_dataset(image_dir, labels_file)
    df.to_csv('face_shape_features.csv', index=False)
except ValueError as e:
    print(e)
    exit(1)

# Load dataset
data = pd.read_csv('face_shape_features.csv')
X = data.drop('label', axis=1)
y = data['label']

# Convert DataFrame to NumPy array
X_np = X.to_numpy()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_np, y, test_size=0.2, random_state=42)

# Set up parameter grid for RandomForest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'class_weight': ['balanced', None]
}

# Set up and run RandomizedSearchCV
clf = RandomForestClassifier(random_state=42)
grid_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=10, cv=5, verbose=2, random_state=42)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

# Train model with best parameters
best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)

# Save the model
joblib.dump(best_clf, './models/face_shape_model.pkl')

# Predict and evaluate
y_pred = best_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=best_clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Load trained model
model = joblib.load('./models/face_shape_model.pkl')

def predict_face_shape(image_path):
    features, has_face = extract_features(image_path)
    if not has_face:
        return "No face detected"
    
    features = np.array(features)
    
    if features.ndim == 1:
        features = features.reshape(1, -1)  # Ensure features have the correct shape for a single prediction
    
    predictions = model.predict(features)
    
    if len(predictions) == 0:
        return "No face detected"
    
    most_common_prediction = Counter(predictions).most_common(1)[0][0]
    
    return most_common_prediction

# Example usage with an image in one of the subfolders
test_image_path = 'test.jpg'  # Adjust this path as needed
face_shape = predict_face_shape(test_image_path)
print(f"Predicted Face Shape: {face_shape}")
