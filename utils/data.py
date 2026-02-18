from tensorflow.keras.datasets import cifar10

def load_data():
    
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # already in the correct order
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    return X_train, y_train, X_test, y_test, class_names
