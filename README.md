# DeepFake Video Detector

A modern deep learning application to detect DeepFake videos using advanced frame-level and temporal analysis. Features a user-friendly Streamlit web app for real-time video analysis and instant feedback.

---

## ✨ Features
- **Upload & Analyze:** Instantly upload a video and receive a DeepFake prediction.
- **Hybrid Model:** Combines EfficientNet (for spatial features) and LSTM (for temporal sequence modeling).
- **Interactive UI:** Clean, modern Streamlit interface with sidebar info, video preview, and color-coded results.
- **Confidence Score:** Visual feedback on model certainty.
- **Easy Deployment:** Run locally with minimal setup.

---

## 🛠️ Tools & Technologies
- Python
- TensorFlow & Keras
- OpenCV
- Streamlit
- NumPy
- scikit-learn

---

## ⚡ Quickstart

1. **Clone the repository**
2. **Install dependencies** (recommended: use a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the model** (optional):
   ```bash
   python train.py
   ```
   This will generate `best_model.h5`.
4. **Run the Streamlit app:**
   ```bash
   streamlit run deepfake_streamlit_app.py
   ```
5. **Open your browser** to the local URL provided by Streamlit (usually http://localhost:8501).

---

## 📦 Project Structure
```
DeepFake Detector Improve/
├── train.py                  # Model training script
├── model/model.py            # Model architecture (EfficientNet + LSTM)
├── deepfake_streamlit_app.py # Streamlit web app
├── frames/                   # Extracted video frames
├── balanced_dataset/         # Balanced dataset for training
├── best_model.h5             # Trained model weights
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

---

## 💡 Usage
- Upload a short video (at least 5 frames, formats: mp4, avi, mov).
- The app will display the video, analyze it, and show whether it is a DeepFake or not, along with a confidence score.

---

## 🧩 Troubleshooting
- **App always predicts 0.5:** This means the model is not trained or not learning. Retrain with more data or check your labels.
- **Model loading errors:** Ensure your TensorFlow/Keras version matches the one used for training. Python 3.8–3.10 is recommended.
- **Video not displaying:** Make sure you upload a supported format (mp4, avi, mov).

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License
This project is licensed under the MIT License.
