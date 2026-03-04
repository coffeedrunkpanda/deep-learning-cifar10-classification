import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from typing import Callable, Tuple, Optional

from utils.config import ExperimentConfig
from utils.augmentation import data_augmentation

def load_data():
    """
    Load the CIFAR-10 dataset from Keras.
    Returns:
        tuple: A tuple containing:
            - X_train (ndarray): Training images with shape (50000, 32, 32, 3)
            - y_train (ndarray): Training labels with shape (50000, 1)
            - X_test (ndarray): Test images with shape (10000, 32, 32, 3)
            - y_test (ndarray): Test labels with shape (10000, 1)
            - class_names (list): List of 10 class names corresponding to CIFAR-10 categories
    """
    
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
    Builds and preprocesses TensorFlow data pipelines for training, validation, and testing.
    Splits the training data into train/validation sets, applies shuffling, preprocessing,
    and batching through a configurable pipeline. Data augmentation is applied only to the
    training dataset.
    Args:
        X_train: Training input features (numpy array or tensor).
        y_train: Training labels (numpy array or tensor, not one-hot encoded).
        X_test: Test input features (numpy array or tensor).
        y_test: Test labels (numpy array or tensor).
        config (ExperimentConfig): Configuration object.
        preprocess_fn (Optional[Callable]): Optional custom preprocessing function to apply
            to each sample. Default is None.
        val_split (float): Fraction of training data to use for validation. Default is 0.2.
        seed (int): Random seed for reproducibility in shuffling. Default is 42.
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: 
            - train_ds: Processed training dataset with augmentation enabled.
            - val_ds: Processed validation dataset with augmentation disabled.
            - test_ds: Processed test dataset with augmentation disabled.
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
    """
    Process a batch of images with resizing, augmentation, and normalization.
    Args:
        image: Input image tensor to be processed.
        label: Label associated with the image.
        preprocess_fn (Optional[Callable]): Optional preprocessing function for transfer learning.
            If provided, applies model-specific preprocessing normalization. Defaults to None.
        normalize (bool): Whether to apply default normalization by dividing pixel values by 255.0.
            Defaults to True.
        new_size (tuple): Target size for image resizing as (height, width, channels).
            Defaults to (96, 96, 3).
        augment (bool): Whether to apply data augmentation during processing.
            Defaults to True.
    Returns:
        tuple: A tuple containing:
            - image: Processed image tensor.
            - label: Original label associated with the image.
    """    

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


def process_pipeline(ds: tf.data.Dataset,
                     batch_size:int,
                     new_size:tuple[int,int,int],
                     preprocess_fn: Optional[Callable] = None,
                     normalize: bool = True,
                     augment:bool = True):
    """
    Process a TensorFlow dataset through a batch processing pipeline.

    Applies batching, image preprocessing, normalization, and optional augmentation
    to the input dataset with automatic tuning and prefetching for optimal performance.

    Args:
        ds (tf.data.Dataset): The input dataset containing images and labels.
        batch_size (int): The number of samples per batch.
        new_size (tuple[int, int, int]): Target image size as (height, width, channels).
        preprocess_fn (Optional[Callable]): Custom preprocessing function to apply to images.
            If None, no custom preprocessing is applied. Defaults to None.
        normalize (bool): Whether to normalize pixel values. Defaults to True.
        augment (bool): Whether to apply data augmentation. Defaults to True.

    Returns:
        tf.data.Dataset: A processed dataset with batched, preprocessed, and optionally
            augmented images, optimized with automatic tuning and prefetching.
    """

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
