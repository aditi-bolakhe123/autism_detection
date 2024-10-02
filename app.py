import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model_path = 'LRCN_model.h5'
model = tf.keras.models.load_model(model_path)

# Define the class names
class_names = ["face padding", "rolling back and forth", "rubbing the skin"]

# Function to preprocess the video
def preprocess_video(video_path, img_size=(64, 64), max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, img_size)
        frames.append(frame)

    cap.release()
    cv2.destroyAllWindows()

    # If fewer frames than max_frames, pad with zeros
    while len(frames) < max_frames:
        frames.append(np.zeros((img_size[0], img_size[1], 3)))

    frames = np.array(frames) / 255.0
    frames = np.expand_dims(frames, axis=0)

    return frames

# Streamlit app
st.title("Video Action Classification")

# File uploader for video
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video("temp_video.mp4")

    # Preprocess the video
    frames = preprocess_video("temp_video.mp4", img_size=(64, 64))

    # Check if the frames are in the expected shape
    st.write(f'Input shape: {frames.shape}')

    # Predict the action
    predictions = model.predict(frames)
    predicted_class_index = np.argmax(predictions, axis=-1)[0]
    predicted_class_name = class_names[predicted_class_index]

    st.write(f'Predicted action class index: {predicted_class_index}')
    st.write(f'Predicted action class name: {predicted_class_name}')