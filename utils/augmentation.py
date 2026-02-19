from tensorflow.keras import Sequential, layers

seed = 42

# data_augmentation = Sequential([
#     layers.RandomFlip(mode = "horizontal", seed = seed),
#     layers.RandomRotation(0.1, fill_mode = "constant"),
#     layers.RandomZoom(0.1),
# ])

# needs to be 0-255
data_augmentation = Sequential([
    layers.RandomFlip(mode = "horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
    layers.RandomBrightness(0.05),
    layers.RandomContrast(0.05),
])