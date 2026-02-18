import tensorflow as tf
from tensorflow.keras.datasets import cifar10

from utils.config import ExperimentConfig
from typing import Callable, Tuple, Optional

def load_data():
    
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # already in the correct order
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    return X_train, y_train, X_test, y_test, class_names


def build_datasets(
    X_train, y_train,
    X_test, y_test,
    preprocess_fn: Optional[Callable] = None,
    val_split: float = 0.2,
    seed: int = 42
):
    """
    Splits, shuffles, preprocesses and batches data into
    train/val/test tf.data pipelines.
    """

    dataset_size = len(X_train)
    train_size = int((1 - val_split) * dataset_size)

    # 1. Build and shuffle full dataset before splitting
    full_dataset = (
        tf.data.Dataset
        .from_tensor_slices((X_train, y_train))  # integers, no one-hot
        .shuffle(buffer_size=1000, seed=seed)
    )

    train_dataset = full_dataset.take(train_size)
    val_dataset   = full_dataset.skip(train_size)
    test_dataset  = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    return train_dataset, val_dataset, test_dataset

# def build_dataset(X_train, y_train, 
#                   X_test, y_test,
#                   preprocess_fn = Optional[Callable] = None,
#                   val_split: float = 0.2,
#                   seed:int = 13
#                   ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:

#     dataset_size = len(X_train)
#     train_size = int((1 - val_split) * dataset_size)

#     # 1. Build and shuffle full dataset before splitting
#     full_dataset = (
#         tf.data.Dataset
#         .from_tensor_slices((X_train, y_train))  # integers, no one-hot
#         .shuffle(buffer_size=1000, seed=seed)
#     )

#     train_dataset = full_dataset.take(train_size)
#     val_dataset = full_dataset.skip(train_size)
#     test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

#     return train_dataset, val_dataset, test_dataset

# def process_batch(image, label, new_size = (96, 96)):
#     image = tf.image.resize(image, new_size)
#     image = preprocess_input(image) 
#     return image, label

# BATCH_SIZE = 32

# # 3. Apply your batching/preprocessing pipeline to BOTH
# def process_pipeline(ds):
#     return (
#         ds
#         .batch(BATCH_SIZE)
#         .map(process_batch, num_parallel_calls=tf.data.AUTOTUNE)
#         .prefetch(tf.data.AUTOTUNE)
#     )

# train_ds = process_pipeline(train_dataset)
# val_ds = process_pipeline(val_dataset)
# test_ds = process_pipeline(test_dataset)
