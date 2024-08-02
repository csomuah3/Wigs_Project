from flask import Flask, request, render_template, send_from_directory, url_for
import os
import joblib
import numpy as np
import cv2
import dlib


app = Flask(__name__)
model = joblib.load('./models/face_shape_model.pkl')

# Initialize face detector and shape predictor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#pretrained model 
predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

#extracting the face features
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

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
    
    return features

def classify_face_shape(image):
    features = extract_features(image)
    if not features:
        return None
    
    features = np.array(features)
    prediction = model.predict(features)
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    face_shape = None
    wig_images = []
    image_path = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            upload_dir = 'static/uploads'
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            file_path = os.path.join(upload_dir, file.filename)
            try:
                file.save(file_path)
            except Exception as e:
                print(f"Error saving file: {e}")
                return f"Error saving file: {e}"

            image = cv2.imread(file_path)
            face_shape = classify_face_shape(image)

            if face_shape:
                print(f"Detected face shape: {face_shape}")

                # Define the directory for wig images (no subdirectories now)
                wig_dir = 'static/wigs_images'

                if os.path.exists(wig_dir):
                    for filename in os.listdir(wig_dir):
                        if filename.endswith(('jpg', 'jpeg', 'png')):
                            # Check if the filename contains the face shape
                            if face_shape.lower() in filename.lower():
                                wig_images.append(os.path.join('wigs_images', filename))
                                print(f"Adding wig image: {os.path.join(wig_dir, filename)}")

            # Serve the uploaded image for preview
            image_path = url_for('static', filename='uploads/' + file.filename)
    
    return render_template('index.html', face_shape=face_shape, wig_images=wig_images, image_path=image_path)

@app.route('/uploads/<filename>')
def serve_uploaded_image(filename):
    return send_from_directory('static/uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)
