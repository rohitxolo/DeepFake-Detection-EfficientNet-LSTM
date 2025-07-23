import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from model.model import build_model

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Config
FRAME_HEIGHT, FRAME_WIDTH = 112, 112
CHANNELS = 3
TIMESTEPS = 5
BATCH_SIZE = 4
EPOCHS = 10
FRAME_DIR = "frames"

# Helper to group frames by video id
from collections import defaultdict

def get_video_groups(frame_dir, prefix):
    groups = defaultdict(list)
    for fname in os.listdir(frame_dir):
        if fname.startswith(prefix) and fname.endswith('.jpg'):
            # e.g., real_00299_f18.jpg -> real_00299
            base = fname.rsplit('_f', 1)[0]
            groups[base].append(fname)
    # Sort frames within each group by frame number
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: int(x.rsplit('_f', 1)[1].split('.')[0]))
    return groups

def load_frame_sequence(frame_dir, frame_files):
    frames = []
    for fname in frame_files:
        img_path = os.path.join(frame_dir, fname)
        img = load_img(img_path, target_size=(FRAME_HEIGHT, FRAME_WIDTH))
        img = img_to_array(img)
        frames.append(img)
    frames = np.array(frames, dtype="float32") / 255.0
    return frames

def load_dataset():
    X = []
    y = []
    # Real
    real_dir = os.path.join(FRAME_DIR, "real")
    real_groups = get_video_groups(real_dir, "real_")
    for video_id, frame_list in real_groups.items():
        if len(frame_list) >= TIMESTEPS:
            # Take first TIMESTEPS frames (or sample randomly for more variety)
            selected = frame_list[:TIMESTEPS]
            frames = load_frame_sequence(real_dir, selected)
            if frames.shape[0] == TIMESTEPS:
                X.append(frames)
                y.append(0)
    # Fake
    fake_dir = os.path.join(FRAME_DIR, "fake")
    fake_groups = get_video_groups(fake_dir, "fake_")
    for video_id, frame_list in fake_groups.items():
        if len(frame_list) >= TIMESTEPS:
            selected = frame_list[:TIMESTEPS]
            frames = load_frame_sequence(fake_dir, selected)
            if frames.shape[0] == TIMESTEPS:
                X.append(frames)
                y.append(1)
    return np.array(X), np.array(y)

# Load and preprocess data
print("[INFO] Loading dataset...")
X, y = load_dataset()
print(f"[INFO] Dataset shape: {X.shape}, Labels shape: {y.shape}")

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Handle class imbalance
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Build model
print("[INFO] Building model...")
model = build_model(FRAME_HEIGHT, FRAME_WIDTH, CHANNELS, TIMESTEPS)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train
print("[INFO] Training started...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights_dict,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

print("[INFO] Training completed.")
