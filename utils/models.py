import tensorflow as tf
from tensorflow.keras.applications import ConvNeXtTiny
from utils.config import ExperimentConfig
from tensorflow.keras import layers, models


def baseline_cnn(config:ExperimentConfig, n_labels:int):
    """Simple 3-layer CNN with batch normalization"""
    conv_kernel_size = (3,3) 
    dropout = 0.5 
    
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, conv_kernel_size, activation='relu', input_shape=config.input_shape),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, conv_kernel_size, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(n_labels, activation='softmax')
    ])

    return model

def simple_cnn(config:ExperimentConfig, n_labels:int):
    """Simple 3-layer CNN without batch normalization"""

    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=config.input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(n_labels, activation='softmax')
    ])

    return model

def deeper_cnn(config:ExperimentConfig, n_labels:int):
    """Simple 4-layer CNN with batch normalization"""
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=config.input_shape),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(n_labels, activation='softmax')
    ])

    return model

def conv_next_tiny(config:ExperimentConfig, n_labels:int):

    base_model = ConvNeXtTiny(
        include_top=False,
        include_preprocessing=True,
        weights="imagenet",
        input_shape=config.input_shape
    )
    base_model.trainable = False  # Freeze base

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(), # Pool spatial features into a vector
        layers.Dense(128, activation='relu'), # Fully connected layer to learn new patterns
        layers.Dense(n_labels, activation='softmax') # Output layer with n_labels classes
    ])



MODELS = {
    "baseline_cnn": baseline_cnn,
    "deeper_cnn": deeper_cnn,
    "simple_cnn": simple_cnn,
    "ConvNeXtTiny": conv_next_tiny
}
