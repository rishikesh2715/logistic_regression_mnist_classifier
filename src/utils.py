import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import cv2

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


def load_elegans_train(image_size=(28, 28), val_split=0.1):
    """
    Load the elegans training dataset and split into train/val sets.
    - Folder: ../data/elegans/0_train -> no worm
    - Folder: ../data/elegans/1_train -> worm

    :param image_size: Image shape (width, height)
    :param val_split: Fraction of training data to use as validation
    :return: train_data, train_labels, val_data, val_labels
    """
    data_dir = "../data/elegans"
    images = []
    labels = []

    print("Loading ELEGANS training data...")
    for label in ["0_train", "1_train"]:
        folder = os.path.join(data_dir, label)
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, image_size)
            img = cv2.equalizeHist(img)  # Histogram equalization
            img = img.astype(np.float32) / 255.0
            images.append(img.flatten())
            labels.append(int(label.split('_')[0]))  # Get 0 or 1

    images = np.array(images)
    labels = np.array(labels)
    print(f"ELEGANS Training data loaded with {len(images)} images.")

    # Shuffle before split
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    images = images[indices]
    labels = labels[indices]

    # Split train/val
    val_size = int(val_split * len(images))
    val_data = images[:val_size]
    val_labels = labels[:val_size]
    train_data = images[val_size:]
    train_labels = labels[val_size:]

    train_labels = one_hot_encode(train_labels, num_classes=2)
    val_labels = one_hot_encode(val_labels, num_classes=2)

    return train_data, train_labels, val_data, val_labels


def load_elegans_test(image_size=(28, 28), return_filenames=False):
    """
    Load the elegans test dataset.
    - Folder: ../data/elegans/0_test -> no worm
    - Folder: ../data/elegans/1_test -> worm

    :param image_size: Image shape (width, height)
    :return: test_data, test_labels
    """
    data_dir = "../data/elegans"
    images = []
    labels = []
    filenames = []

    print("Loading ELEGANS testing data...")
    for label in ["0_test", "1_test"]:
        folder = os.path.join(data_dir, label)
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, image_size)
            img = cv2.equalizeHist(img)  # Histogram equalization
            img = img.astype(np.float32) / 255.0
            images.append(img.flatten())
            labels.append(int(label.split('_')[0]))  # Get 0 or 1
            if return_filenames:
                filenames.append(filename)

    images = np.array(images)
    labels = np.array(labels)
    print(f"ELEGANS Testing data loaded with {len(images)} images.")

    test_labels = one_hot_encode(labels, num_classes=2)

    if return_filenames:
        return images, test_labels, filenames
    return images, test_labels


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
    print(f"MNIST Training data loaded with {len(data)} images.")

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
    labels = read_idx_labels('../data/MNIST/t10k-labels.idx1-ubyte')
    assert len(data) == len(labels)
    print(f"MNIST Testing data loaded with {len(data)} images.")

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


def plot_results(train_losses, val_losses, train_accuracies, val_accuracies, title_suffix=''):
    """
    Plot learning curves:
    -- Training loss vs Validation loss
    -- Training accuracy vs Validation accuracy
    -- We will plot the confusio matrix in evaluate.py
    -- Inference with test images should also be done in evaluate.py

    :param train_losses: List of training losses
    :param val_losses: List of validation losses
    :param train_accuracies: List of training accuracies
    :param val_accuracies: List of validation accuracies
    :return: None
    """

    epochs = list(range(len(train_losses)))

    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve - {title_suffix}')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy Curve - {title_suffix}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'../outputs/loss_accuracy_{title_suffix}.png')
    plt.show()


if __name__ == '__main__':
    # Test the MNIST loading and plotting functions
    train_data, train_label, _, _ = load_mnist_train()
    plot_sample_images(train_data, train_label)

    # Test the ELEGANS loading and plotting functions
    # First plot is train MNIST data
    # Second plot is validation MNIST data
    # Third plot is test MNIST data
    train_data, train_labels, val_data, val_labels = load_elegans_train()
    test_data, test_labels = load_elegans_test()
    plot_sample_images(train_data, train_labels, num_images=10)
    plot_sample_images(val_data, val_labels, num_images=10) 
    plot_sample_images(test_data, test_labels, num_images=10)
