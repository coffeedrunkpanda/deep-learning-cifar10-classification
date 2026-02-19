import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from typing import Callable, Tuple, Optional

from utils.config import ExperimentConfig
from utils.augmentation import data_augmentation

def load_data():
    
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # already in the correct order
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    return X_train, y_train, X_test, y_test, class_names


def build_datasets(
    X_train, y_train,
    X_test, y_test,
    config:ExperimentConfig,
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


    full_dataset = (
        tf.data.Dataset
        .from_tensor_slices((X_train, y_train))  # integers, no one-hot
        .shuffle(buffer_size=1000, seed=seed)
    )

    train_dataset = full_dataset.take(train_size)
    val_dataset   = full_dataset.skip(train_size)
    test_dataset  = tf.data.Dataset.from_tensor_slices((X_test, y_test))


    train_ds = process_pipeline(train_dataset,
                                batch_size=config.batch_size,
                                new_size=config.input_shape,
                                preprocess_fn=preprocess_fn,
                                normalize=config.normalize,
                                augment=config.augment)
    
    # data augmentation is disabled for validation
    val_ds = process_pipeline(val_dataset,
                              batch_size=config.batch_size,
                              new_size=config.input_shape,
                              preprocess_fn=preprocess_fn,
                              normalize=config.normalize,
                              augment=False)

    # data augmentation is disabled for testing
    test_ds = process_pipeline(test_dataset,
                               batch_size=config.batch_size,
                               new_size=config.input_shape,
                               preprocess_fn=preprocess_fn,
                               normalize=config.normalize,
                               augment=False)
    
    return train_ds, val_ds, test_ds


def process_batch(image, label,
                  preprocess_fn: Optional[Callable] = None,
                  normalize = True,
                  new_size = (96, 96, 3),
                  augment:bool = True):
    
    image = tf.image.resize(image, (new_size[0], new_size[1]))

    if augment:
        image = data_augmentation(image, training=True)

    # Transfer learning normalization and preprocesing    
    if preprocess_fn:
        image = preprocess_fn(image) 
    
    if normalize:
        # Default normalization
        image = image/255.0
        
    return image, label

# Apply your batching/preprocessing pipeline to BOTH
def process_pipeline(ds: tf.data.Dataset,
                     batch_size:int,
                     new_size:tuple[int,int,int],
                     preprocess_fn: Optional[Callable] = None,
                     normalize: bool = True,
                     augment:bool = True):

    return (
        ds
        .batch(batch_size)
        .map(
            lambda image, label: process_batch(
                image,label, preprocess_fn, normalize, new_size, augment
                ),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
