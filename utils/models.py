import tensorflow as tf

def baseline_cnn(input_shape:tuple[int,int,int] =(32, 32, 3),
                 n_labels:int = 10):
    """Simple 3-layer CNN with batch normalization"""
    conv_kernel_size = (3,3) 
    dropout = 0.5 
    
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, conv_kernel_size, activation='relu', input_shape=input_shape),
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

def simple_cnn(input_shape:tuple =(32, 32, 3), n_labels:int = 10):
    """Simple 3-layer CNN without batch normalization"""

    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(n_labels, activation='softmax')
    ])

    return model

def deeper_cnn(input_shape:tuple =(32, 32, 3), n_labels:int = 10):
    """Simple 4-layer CNN with batch normalization"""
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
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

MODELS = {
    "baseline_cnn": baseline_cnn,
    "deeper_cnn": deeper_cnn,
    "simple_cnn": simple_cnn
}
