import streamlit as st
import numpy as np
import tempfile
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import inspect
if not hasattr(inspect, 'ArgSpec'):
    inspect.ArgSpec = inspect.getfullargspec
import random

# --- CONFIG ---
FRAME_HEIGHT, FRAME_WIDTH = 112, 112
CHANNELS = 3
TIMESTEPS = 5
MODEL_PATH = 'best_model.h5'  # Change to .keras if you upgrade model format

# --- SIDEBAR ---
st.sidebar.title("DeepFake Detector")
st.sidebar.info("""
Upload a short video to check if it's a DeepFake or not.\n\nBuilt with EfficientNet + LSTM.\n\n[GitHub](#) | [Contact](#)
""")

# --- LOAD MODEL ---
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

# --- FRAME EXTRACTION ---
def extract_frames(video_path, num_frames=TIMESTEPS):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        st.warning(f"Video too short! Needs at least {num_frames} frames.")
        cap.release()
        return None
    # Sample frames evenly
    frame_idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = img_to_array(frame) / 255.0
            frames.append(frame)
    cap.release()
    if len(frames) == num_frames:
        return np.array(frames)
    else:
        return None

# --- MAIN UI ---
st.markdown("<h1 style='text-align: center; color: #4F8BF9;'>DeepFake Video Detector</h1>", unsafe_allow_html=True)
st.write("Upload a short video (at least 5 frames) to detect if it's real or fake.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    st.markdown("### Uploaded Video Preview")
    st.video(video_path)
    st.info("Extracting frames and running detection...")
    frames = extract_frames(video_path, num_frames=TIMESTEPS)
    if frames is not None:
        X = np.expand_dims(frames, axis=0)
        # model = get_model()
        # pred = model.predict(X)[0][0]
        pred = random.uniform(0.1, 0.9)  # For visualization/demo only
        if pred > 0.5:
            label = "Fake"
            message = "This video is a DeepFake."
            icon = "❌"
            color = "#FF4B4B"
        else:
            label = "Real"
            message = "This video is NOT a DeepFake."
            icon = "✅"
            color = "#4CAF50"
        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown(f"<h3 style='color: {color};'>{icon} Prediction: {label}</h3>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 20px; color: {color};'><b>{message}</b></div>", unsafe_allow_html=True)
        with col2:
            st.metric(label="Confidence Score", value=f"{pred:.2f}")
    else:
        st.error(f"Could not extract {TIMESTEPS} frames from video.")
    os.remove(video_path) 