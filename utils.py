import os
import cv2
from tqdm import tqdm

def extract_frames(video_dir, output_dir, label):
    os.makedirs(output_dir, exist_ok=True)
    videos = os.listdir(video_dir)

    for vid in tqdm(videos, desc=f"Extracting from {label} videos"):
        if not vid.endswith('.mp4'):
            continue
        cap = cv2.VideoCapture(os.path.join(video_dir, vid))
        frame_num = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % 25 == 0:  # capture 1 frame/sec assuming 25fps
                frame = cv2.resize(frame, (224, 224))
                filename = f"{label}_{vid[:-4]}_f{saved}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), frame)
                saved += 1
            frame_num += 1
        cap.release()

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def load_video_frames(video_path, num_frames=30, target_size=(64, 64)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Edge case: skip if no frames
    if total_frames <= 0:
        cap.release()
        return []

    num_frames = int(num_frames)

    # Ensure total_frames is a valid int (prevent dtype=object errors)
    total_frames = int(total_frames)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = img_to_array(frame)
        frames.append(frame)

    cap.release()
    return np.array(frames, dtype="float32") / 255.0  # Normalize pixel values


import numpy as np
from sklearn.utils import shuffle

def load_dataset(real_dir="dataset/real", fake_dir="dataset/fake", num_frames=30, target_size=(64, 64)):
    X = []
    y = []

    print("[INFO] Loading real videos...")
    for filename in os.listdir(real_dir):
        if filename.endswith(".mp4"):
            path = os.path.join(real_dir, filename)
            frames = load_video_frames(path, num_frames=num_frames, target_size=target_size)
            if len(frames) == num_frames:
                X.append(frames)
                y.append(0)  # Label 0 for real

    print("[INFO] Loading fake videos...")
    for filename in os.listdir(fake_dir):
        if filename.endswith(".mp4"):
            path = os.path.join(fake_dir, filename)
            frames = load_video_frames(path, num_frames=num_frames, target_size=target_size)
            if len(frames) == num_frames:
                X.append(frames)
                y.append(1)  # Label 1 for fake

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int")

    X, y = shuffle(X, y, random_state=42)
    return X, y
