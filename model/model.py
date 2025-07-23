import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

def build_model(frame_height=224, frame_width=224, channels=3, timesteps=10):
    """
    Returns a model that takes a sequence of frames (video) and predicts real or fake.
    Input shape: (batch_size, timesteps, height, width, channels)
    """

    # Input for sequence of frames
    video_input = layers.Input(shape=(timesteps, frame_height, frame_width, channels))

    # EfficientNetB0 as feature extractor (no top layer)
    cnn_base = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')
    cnn_base.trainable = False  # Freeze for transfer learning

    # Apply CNN to each frame in the sequence using TimeDistributed
    cnn_features = layers.TimeDistributed(cnn_base)(video_input)

    # LSTM to capture temporal features
    lstm = layers.LSTM(256, return_sequences=False, dropout=0.3)(cnn_features)

    # Classification
    output = layers.Dense(1, activation='sigmoid')(lstm)

    # Define model
    model = models.Model(inputs=video_input, outputs=output)

    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
