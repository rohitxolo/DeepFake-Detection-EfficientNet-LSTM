from utils import extract_frames

if __name__ == "__main__":
    extract_frames("balanced_dataset/real", "frames/real", "real")
    extract_frames("balanced_dataset/fake", "frames/fake", "fake")
