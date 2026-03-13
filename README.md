# 🚀 Deep Learning Image Classification on CIFAR-10

This repository contains an end-to-end Machine Learning pipeline for image classification using Convolutional Neural Networks (CNNs) built with TensorFlow and Keras. 

This project was developed as part of a Data Science & Machine Learning bootcamp to explore different neural network architectures, efficient data pipelines, and experiment tracking.


## 🧠 Educational Concepts Covered
If you are learning DS/ML, this repository serves as a great bridge between beginner concepts and advanced implementations:
- **`tf.data` Pipelines**: Moving away from holding all data in memory, this project uses highly optimized TensorFlow data pipelines with `.prefetch()` and `.AUTOTUNE`.
- **Data Augmentation**: Implementing augmentation inside the model graph via `keras.Sequential` layers to prevent overfitting and make the model robust.
- **Batch Normalization**: Using standardizations between layers to prevent exploding/vanishing gradients, allowing for deeper networks and faster convergence.
- **Transfer Learning**: Utilizing pre-trained computer vision models (MobileNetV2, ConvNeXtTiny) and freezing their weights to learn new classification layers on minimal data.
- **Experiment Tracking (MLOps)**: Using Weights & Biases (`wandb`) to automatically log hyperparameters, validation metrics, early stopping, and even image prediction samples.

---

## 📂 Project Structure

- **`config.py`**: Contains `ExperimentConfig` and `WandbConfig` dataclasses. This is a clean way to manage hyperparameters (learning rate, batch size, input shape) without hardcoding them.
- **`data.py`**: Handles loading the CIFAR-10 dataset and applying the `tf.data` pipeline. It manages train/val/test splits, resizing, normalization, and conditionally applies augmentation.
- **`augmentation.py`**: Defines a sequential Keras model for on-the-fly image transformations (flips, rotations, contrast adjustments).
- **`models.py`**: Contains the dictionary of available architectures, ranging from a custom 3-layer `baseline_cnn` to the advanced `GuaxinimCNN` and transfer-learning base models.
- **`train.py`**: The core training loop. Initializes W&B runs, compiles the model, and utilizes `EarlyStopping` callbacks. Saves the best model locally and uploads it as an artifact.
- **`eval.py`**: Provides evaluation utilities like classification reports, confusion matrix heatmaps (via seaborn), and plotting functions for correct/incorrect visual predictions.

---

## 📊 The Dataset: CIFAR-10
The dataset consists of 60,000 color images (32x32 pixels, 3 channels) split across 10 classes:
*Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.*
- **Training Set**: 50,000 images (further split to create a 20% validation set internally).
- **Test Set**: 10,000 images held out completely for final evaluation.

---

## 🏗️ Model Architectures

The framework allows for easily swapping between different neural network experiments:

1. **Custom Deep CNNs**:
   - `simple_cnn`: Baseline CNN with max pooling and dense layers.
   - `baseline_cnn`: Adds Batch Normalization to the simple architecture.
   - `deeper_cnn`: A 4-block convolution architecture with heavy batch normalization and a 0.5 Dropout layer.
   - `GuaxinimCNN` (Raccoon CNN): A highly custom 6-block CNN demonstrating complex filter sizing and staged dropout layers.

2. **Transfer Learning Models**:
   - `MobileNetV2`: Optimized for edge devices with a smaller parameter footprint.
   - `ConvNeXtTiny`: Modern CNN architecture attempting to rival Vision Transformers.

---

### 🏆 Model Performance and Trade-Offs

Comparing the top performers from each architectural approach reveals the clear trade-offs between computational cost, input resolution, and predictive power. Transfer learning heavily outpaces custom architectures in accuracy, while lightweight models like MobileNetV2 offer the best return on investment for training time.

| Model Category | Architecture | Input Shape | Val Accuracy | Runtime | Architectural Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Transfer Learning (Heavy)** | `ConvNeXtTiny` | 224x224 | **92.6%** | ~37 min | Highest overall accuracy; heavily relies on upscaling images to 224x224 for maximum feature extraction . |
| **Transfer Learning (Edge)** | `MobileNetV2` | 96x96 | **85.8%** | ~1.3 min | The efficiency sweet spot; achieves excellent accuracy in a fraction of the time, perfect for edge devices  . |
| **Custom (Optimized)** | `deeper_cnn` | 64x64 | **77.4%** | ~1.6 min | Best model built from scratch; upscaling to 64x64 prevents the 4 pooling layers from destroying spatial geometry  . |
| **Custom (Experimental)** | `GuaxinimCNN` | 32x32 | **76.3%** | ~2.8 min | "Raccoon CNN" proves that staggered dropout and minimal pooling can still yield strong results on native 32x32 images  . |
| **Custom (Baseline)** | `baseline_cnn` | 32x32 | **74.0%** | ~1.5 min | Standard 3-layer CNN with Batch Normalization serving as the benchmark for all other experiments . |

**Dashboards**: 
- [W&B Dashboard for Custom CNNs](https://wandb.ai/coffeedrunk/deep-learning-cifar10-classification_custom/workspace?nw=nwusercoffeedrunk) 
- [W&B Dashboard for Fine-Tuned Models](https://wandb.ai/coffeedrunk/deep-learning-cifar10-classification_final/workspace?nw=nwusercoffeedrunk)

## 📈 Conclusions
From tracking our metrics across different setups, our team observed:

Pooling vs Image Size: If you use Max Pooling extensively, you need larger input image sizes (upscaling 32x32 to 64x64 or 96x96) so the spatial dimensions don't shrink to zero too early.

Regularization Techniques: It is possible to achieve similar regularization results combining standard Dropout with minimal pooling.

Normalization: Default pixel normalization (dividing by 255) is standard, but doesn't universally guarantee higher accuracy on this specific small image scale depending on the transfer learning base used.


## 🛠️ Tech Stack
* Languages & Core Libraries: Python, NumPy

* Deep Learning Framework: TensorFlow, Keras

* MLOps & Experiment Tracking: Weights & Biases (W&B)

* Evaluation & Visualization: Scikit-Learn, Matplotlib, Seaborn

## 👤 Authors
Built by me [@coffeedrunkpanda](https://github.com/coffeedrunkpanda), combined with the independent work of [@0906manish](https://github.com/0906manish/), and [@Esalsac](https://github.com/Esalsac/).  
