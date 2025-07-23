import os
import shutil
import random

# Update these paths based on actual folders
REAL_DIRS = ['dataset/Celeb-real', 'dataset/YouTube-real']
FAKE_DIRS = ['dataset/Celeb-synthesis']

BALANCED_REAL_DIR = 'balanced_dataset/real'
BALANCED_FAKE_DIR = 'balanced_dataset/fake'

# Create output directories if not exist
os.makedirs(BALANCED_REAL_DIR, exist_ok=True)
os.makedirs(BALANCED_FAKE_DIR, exist_ok=True)

# Collect all video paths
real_videos = []
for real_dir in REAL_DIRS:
    real_videos += [os.path.join(real_dir, vid) for vid in os.listdir(real_dir)]

fake_videos = []
for fake_dir in FAKE_DIRS:
    fake_videos += [os.path.join(fake_dir, vid) for vid in os.listdir(fake_dir)]

# Shuffle and match counts
min_count = min(len(real_videos), len(fake_videos))
real_videos = random.sample(real_videos, min_count)
fake_videos = random.sample(fake_videos, min_count)

print(f"Balancing {min_count} real and {min_count} fake videos")

# Copy files to balanced_dataset
for i, vid_path in enumerate(real_videos):
    shutil.copy(vid_path, os.path.join(BALANCED_REAL_DIR, f"real_{i}.mp4"))

for i, vid_path in enumerate(fake_videos):
    shutil.copy(vid_path, os.path.join(BALANCED_FAKE_DIR, f"fake_{i}.mp4"))

print("✅ Balanced dataset created.")
