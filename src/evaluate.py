"""
evaluate.py

TODO:

1. Load trained model 

2. Load test data:
   - Use utils.py to load and preprocess test set

3. Evaluate model performance:
   - Run forward pass on test data
   - Get predicted labels using argmax
   - Compute accuracy

4. Generate evaluation results:
   - Build confusion matrix
   - Maybe we should also display/save sample predictions

5. Output results:
   - Write results to Excel file with:
     - Column 1: image filenames
     - Column 2: predicted labels
     - Count of predictions per label
   - Save confusion matrix plot to outputs folder
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import LogisticRegression
import utils
from sklearn.metrics import confusion_matrix
from collections import Counter

def load_test_images():
    data, labels = utils.load_mnist_test()

    X_test = data.T
    Y_test = np.argmax(labels, axis=1)

    n_features = X_test.shape[0]
    n_classes = labels.shape[1]

    return X_test, Y_test, n_features, n_classes


def load_model(model_path, n_features, n_classes):
    model = LogisticRegression(n_features, n_classes)
    model.load_model(model_path)

    return model


def evaluate(model, X_test, Y_test_labels):
    Y_pred_onehot = model.predict(X_test)
    # Y_pred_labels = np.argmax(Y_pred_onehot, axis=0)
    Y_pred_labels = model.predict(X_test)


    accuracy = np.mean(Y_pred_labels == Y_test_labels) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    cm = confusion_matrix(Y_test_labels, Y_pred_labels)
    return Y_pred_labels, accuracy, cm


def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('../outputs/confusion_matrix.png')
    plt.show()


def show_sample_predictions(X_test, Y_true, Y_pred, num_images=25):

    # Pick misclassified & correctly classified indices
    correct_mask = Y_pred == Y_true
    incorrect_mask = ~correct_mask

    # Flattened test images → reshape
    X_test = X_test.T  # (samples, features)
    image_shape = (28, 28)

    # Combine both correct & incorrect for mixed grid
    indices = np.concatenate([
        np.where(incorrect_mask)[0][:num_images // 2],
        np.where(correct_mask)[0][:num_images // 2]
    ])

    num_cols = 5
    num_rows = int(np.ceil(num_images / num_cols))
    plt.figure(figsize=(num_cols * 2, num_rows * 2))

    for i, idx in enumerate(indices):
        img = X_test[idx].reshape(image_shape)
        true_label = Y_true[idx]
        pred_label = Y_pred[idx]
        is_correct = (true_label == pred_label)

        color = 'green' if is_correct else 'red'
        title = f"T:{true_label} P:{pred_label}"

        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title, color=color)
        plt.axis('off')

    plt.tight_layout()
    os.makedirs('../outputs', exist_ok=True)
    plt.savefig('../outputs/sample_predictions.png')
    plt.show()



if __name__ == '__main__':
    model_path = '../models/trained_model.pkl'

    X_test, Y_test_labels, n_features, n_classes = load_test_images()
    model = load_model(model_path, n_features, n_classes)

    Y_pred_labels, accuracy, cm = evaluate(model, X_test, Y_test_labels)
    show_sample_predictions(X_test, Y_test_labels, Y_pred_labels)


    cm = plot_confusion_matrix(cm)
