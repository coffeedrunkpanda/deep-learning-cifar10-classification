import tensorflow as tf
from tensorflow.keras import layers, models

from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


from utils.config import ExperimentConfig


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
    """CNN with 4 convolution blocks and 1 dense layer. Includes batch normalization"""
    model = tf.keras.Sequential([
    # Conv block 1 
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=config.input_shape),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Conv block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Conv block 3
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    
    # Dense layer block
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

    return model

def guaxinim_cansado_cnn(config:ExperimentConfig, n_labels:int):
    """ Guaxinim/Raccoon CNN, its random and it might not make sense but he is still standing"""
    model = tf.keras.Sequential([
    # Conv block 1 
    tf.keras.layers.Conv2D(16, (3, 3), input_shape=config.input_shape, padding="same"),
    tf.keras.layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D((2, 2)),

    # Conv block 2
    tf.keras.layers.Conv2D(16, (3, 3), padding="same"),
    tf.keras.layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.2),
    
    # Conv block 3
    tf.keras.layers.Conv2D(32, (3, 3), padding="same"),
    tf.keras.layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.1),
    
    # Conv block 4
    tf.keras.layers.Conv2D(64, (3, 3)),
    tf.keras.layers.BatchNormalization(),
    layers.ReLU(),

    # Conv block 5
    tf.keras.layers.Conv2D(64, (3, 3)),
    tf.keras.layers.BatchNormalization(),
    layers.ReLU(),

    # Conv block 6
    tf.keras.layers.Conv2D(128, (3, 3)),
    tf.keras.layers.BatchNormalization(),
    layers.ReLU(),
    tf.keras.layers.Flatten(),
    
    # Dense layer block
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(n_labels, activation='softmax')
    ])

    return model

def mobile_net_v2(config:ExperimentConfig, n_labels:int):

    base_model = MobileNetV2(
        include_top=False,
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

    return model


MODELS = {
    "baseline_cnn": baseline_cnn,
    "deeper_cnn": deeper_cnn,
    "simple_cnn": simple_cnn,
    "ConvNeXtTiny": conv_next_tiny,
    "MobileNetV2": mobile_net_v2,
    "GuaxinimCNN": guaxinim_cansado_cnn
}
