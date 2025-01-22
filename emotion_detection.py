import cv2
import os
import numpy as np
import time
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load model
# model = load_model('/Users/muradsiddiqui/Documents/Winter Semester/RT/Project/RT Project Emotion Detection /Model_1.h5')
model = load_model('/Users/muradsiddiqui/Documents/Winter Semester/RT/Project/RT Project Emotion Detection /Model_1new.h5')

# Define emotion labels
emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
label_encoder = LabelEncoder()
label_encoder.fit(emotion_labels)

# Load Haar cascade for face detection
cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Use GPU acceleration for OpenCV (if available)
cv2.setUseOptimized(True)
cv2.setNumThreads(4)  # Allow OpenCV to use multiple threads

video_capture = cv2.VideoCapture(0)

# Initialize variables for label persistence
last_label = "No Face Detected"
last_confidence = 0.0
frame_count = 0  # To optimize face detection frequency
last_detected_time = time.time()  

while True:
    start_time = time.time()  # Start timer for FPS calculation

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1, 1)  # Flip for mirror effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize default values for latency metrics
    face_detection_time = 0
    preprocessing_time = 0
    inference_time = 0

    # Run face detection every 3rd frame (to improve FPS)
    if frame_count % 3 == 0:
        t1 = time.time()
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48), flags=cv2.CASCADE_SCALE_IMAGE)
        face_detection_time = time.time() - t1  # Time taken for face detection
    frame_count += 1

    detected_face = False  # Track if a face is found in this frame

    for (x, y, w, h) in faces:
        detected_face = True  # Mark face detected

        # Measure preprocessing time
        t2 = time.time()
        face_image = gray[y:y+h, x:x+w]  # Crop face
        face_image_48by48 = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_AREA)  # Resize to 48x48
        model_input_image = np.reshape(face_image_48by48, [1, 48, 48, 1])  # Reshape for CNN
        preprocessing_time = time.time() - t2  # Time taken for preprocessing

        # Measure model inference time
        t3 = time.time()
        y_new = model.predict(model_input_image)
        inference_time = time.time() - t3  # Time taken for model prediction

        # Get predicted emotion label
        max_index = np.argmax(y_new, axis=1)
        y_new_label = label_encoder.inverse_transform(max_index)
        confidence = y_new[0][max_index[0]] * 100  # Confidence percentage

        # Only update label if a certain time has passed (cooldown mechanism)
        if time.time() - last_detected_time > 0.5:  # Update every 0.5 seconds
            last_label = y_new_label[0]
            last_confidence = confidence
            last_detected_time = time.time()

        # Draw rectangle + label with confidence
        label_with_confidence = f"{y_new_label[0]}: {confidence:.2f}%"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label_with_confidence, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # If no face detected, keep showing the last detected emotion for 2 seconds
    if not detected_face and time.time() - last_detected_time > 2:
        last_label = "No Face Detected"
        last_confidence = 0.0

    # Measure total frame processing time
    total_latency = time.time() - start_time
    fps = 1 / total_latency  # Calculate FPS

    # Display FPS & Latency on Video Output
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Latency: {total_latency * 1000:.2f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Preprocessing: {preprocessing_time * 1000:.2f} ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Inference: {inference_time * 1000:.2f} ms", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display last detected emotion in top-left (with cooldown)
    cv2.putText(frame, f"Last Detected: {last_label} ({last_confidence:.2f}%)", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Display frame
    cv2.imshow('Real-Time Facial Expression Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
