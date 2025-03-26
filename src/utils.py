import numpy as np
import matplotlib.pyplot as plt

def load_csv(path):
    """
    Load data from a csv file. The first column is the label 
    and the rest are the pixel values from the MNIST dataset.

    The pixel values are normalized by dividing by 255.

    :param path: path to the csv file
    :return: A tuple of numpy arrays (data, labels)
        - data: 2D numpy array of shape (n_samples, n_features)
        - labels: 1D numpy array of shape (n_samples,) containing the labels ( integers in [0, 9] )
    
    """

    data = []
    labels = []
    with open(path, 'r') as f:
        rows = f.readlines()
        for row in rows:
            row = row.strip().split(',')
            data.append([int(x)/255 for x in row[1:]])
            labels.append(int(row[0]))

    return np.array(data), np.array(labels)


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
    data, label = load_csv('../data/mnist_train.csv')
    assert len(data) == len(label)
    print("Training data loaded with {count} images".format(count=len(data)))


    # One-hot encode
    num_samples = len(label)
    num_classes = 10
    one_hot_labels = np.zeros((num_samples, num_classes))

    for i in range(num_samples):
        one_hot_labels[i, label[i]] = 1

    
    # Split the train set in train and validation set (80-20 split)
    split_idx = int(0.8 * num_samples)
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
    data, label = load_csv('../data/mnist_test.csv')
    assert len(data) == len(label)
    print("Testing data loaded with {count} images".format(count=len(data)))

    # one-hot encode the labels
    num_samples = len(label)
    num_classes = 10

    one_hot_labels = np.zeros((num_samples, num_classes))

    for i in range(num_samples):
        one_hot_labels[i, label[i]] = 1

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
