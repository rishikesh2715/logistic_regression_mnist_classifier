import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import cv2

def read_idx_images(path):
    """
    Read the idx file and return the data as a numpy array

    :param path: path to the idx file
    :return: 2D numpy array of shape (n_samples, n_features) containing the images
    """
    with open(path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        image_data = image_data.reshape(num_images, rows * cols).astype(np.float32) / 255.0
    return image_data

def read_idx_labels(path):
    """
    Read the idx file and return the data as a numpy array

    :param path: path to the idx file
    :return: 1D numpy array of shape (n_samples,) containing the labels ( integers in [0, 9] )
    """
    with open(path, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        label_data = np.frombuffer(f.read(), dtype=np.uint8)
    return label_data

def one_hot_encode(labels, num_classes=10):
    """
    One-hot encode the labels

    :param labels: 1D numpy array of shape (n_samples,) containing the labels ( integers in [0, 9] )
    :param num_classes: number of classes

    :return: 2D numpy array of shape (n_samples, num_classes) containing the one-hot encoded labels
    """
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def load_mnist_train():
    """
    Load the MNIST training data

    :return:
        train_data (np.ndarray): Training images as a numpy array
        train_label (np.ndarray): One-hot encoded training labels
        val_data (np.ndarray): Validation images as a numpy array 
        val_label (np.ndarray): One-hot encoded validation labels as a numpy array 
    
    """
    
    print("Loading MNIST training data...")
    data = read_idx_images('../data/MNIST/train-images.idx3-ubyte')
    labels = read_idx_labels('../data/MNIST/train-labels.idx1-ubyte')
    assert len(data) == len(labels)
    print(f"Training data loaded with {len(data)} images.")

    # one-hot encode the labels
    one_hot_labels = one_hot_encode(labels)

    # Split the data into training and validation sets
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    train_label = one_hot_labels[:split_idx]
    val_data = data[split_idx:]
    val_label = one_hot_labels[split_idx:]

    return train_data, train_label, val_data, val_label


def load_mnist_test():
    """
        Load MNIST testing data with labels
        :return:
            test_data (np.ndarray): Testing images as a numpy array
            test_label (np.ndarray): One-hot encoded testing labels as a numpy array 
    """
    # Load testing data
    print("Loading testing data...")
    data = read_idx_images('../data/MNIST/t10k-images.idx3-ubyte')
    labels = read_idx_labels('../data/MNISTt10k-labels.idx1-ubyte')
    assert len(data) == len(labels)
    print(f"Testing data loaded with {len(data)} images.")

    # one-hot encode the labels
    one_hot_labels = one_hot_encode(labels)

    return data, one_hot_labels


def plot_sample_images(data, labels, num_images=10):
    """
    Plots a grid of sample MNIST images with their corresponding labels.

    :param data: numpy array of shape (n_samples, 784) — flattened images
    :param labels: numpy array of shape (n_samples,) or (n_samples, 10) — class labels or one-hot
    :param num_images: number of images to display
    """
    if labels.ndim > 1:
        labels = np.argmax(labels, axis=1)

    indices = np.random.choice(len(data), num_images, replace=False)
    images = data[indices]
    labels = labels[indices]

    cols = 5
    rows = int(np.ceil(num_images / cols))
    plt.figure(figsize=(cols * 2, rows * 2))

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_results():
    """
    Plot whatever curves we need to show that our model has learned
    -- Training loss vs Validation loss
    -- Training accuracy vs Validation accuracy
    -- Any other metric that we want to track
    -- Confusion matrix etc

    """
    ################################################################
    # TODO:
    #    1) Plot learning curves for training and validation loss
    #    2) Plot learning curves for training and validation accuracy
    #    3) Plot confusion matrix
    #    4) Maybe an image or two with true labels and predictions
    ################################################################


if __name__ == '__main__':
    train_data, train_label, _, _ = load_mnist_train()
    plot_sample_images(train_data, train_label)
