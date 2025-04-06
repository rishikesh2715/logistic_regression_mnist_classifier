import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import LogisticRegression
import utils
from sklearn.metrics import confusion_matrix
import argparse
import re

def load_test_images(dataset):
    """
    Load test images and labels from the elegans dataset.

    :param: None
    :return: X_test (features), Y_test (labels), number of features, number of classes
    :rtype: tuple
    """
    if dataset == 'mnist':
        data, labels = utils.load_mnist_test()
        filenames = [f"img_{i:05d}" for i in range(len(data))]
        model_path = '../models/trained_model_mnist.pkl'
    else:
        data, labels, filenames = utils.load_elegans_test(return_filenames=True)
        model_path = '../models/trained_model_elegans.pkl'

    X_test = data.T
    Y_test = np.argmax(labels, axis=1)
    n_features = X_test.shape[0]
    n_classes = labels.shape[1]

    return X_test, Y_test, filenames, n_features, n_classes, model_path


def load_model(model_path, n_features, n_classes):
    """
    Load the trained model from the specified path.
    :param model_path: Path to the trained model file
    :param n_features: Number of features in the input data
    :param n_classes: Number of classes in the output data
    :return: Loaded model
    :rtype: LogisticRegression
    """
    model = LogisticRegression(n_features, n_classes)
    model.load_model(model_path)

    return model


def evaluate(model, X_test, Y_test_labels):
    """
    Evaluate the model on the test set.
    :param model: Trained model
    :param X_test: Test features
    :param Y_test_labels: Test labels
    :return: Predicted labels, accuracy, confusion matrix
    :rtype: tuple
    """
    Y_pred_labels = model.predict(X_test)

    accuracy = np.mean(Y_pred_labels == Y_test_labels) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    cm = confusion_matrix(Y_test_labels, Y_pred_labels)

    # Correct and incorrect predictions
    correct = Y_pred_labels == Y_test_labels
    total_correct = np.sum(correct)
    total_wrong = len(correct) - total_correct
    print(f"\nTotal Correct Predictions: {total_correct}")
    print(f"Total Wrong Predictions  : {total_wrong}")

    # Per-class correct/wrong count
    print("\n=== Per-Class Accuracy ===")
    classes = np.unique(Y_test_labels)
    for c in classes:
        mask = Y_test_labels == c
        correct_c = np.sum((Y_pred_labels[mask] == c))
        total_c = np.sum(mask)
        print(f"Class {c}: {correct_c}/{total_c} correct ({(correct_c / total_c * 100):.2f}%)")

    return Y_pred_labels, accuracy, cm


def plot_confusion_matrix(cm, dataset):
    """
    Plot the confusion matrix using seaborn heatmap.
    :param cm: Confusion matrix
    :return: None
    """
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', square=True, cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset.upper()}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    os.makedirs('../outputs', exist_ok=True)
    plt.savefig(f'../outputs/confusion_matrix_{dataset.lower()}.png')
    plt.show()


def show_sample_predictions(X_test, Y_true, Y_pred, dataset, num_images=25):
    """
    Show sample predictions from the test set.
    :param X_test: Test features
    :param Y_true: True labels
    :param Y_pred: Predicted labels
    :param num_images: Number of images to display
    :return: None
    """
    correct_mask = Y_pred == Y_true
    incorrect_mask = ~correct_mask

    X_test = X_test.T  # (samples, features)
    image_size = int(np.sqrt(X_test.shape[1]))  
    image_shape = (image_size, image_size)

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
    plt.savefig(f'../outputs/sample_predictions_{dataset.lower()}.png')
    plt.show()


def save_predictions_to_excel(filenames, Y_pred, dataset):
    os.makedirs('../outputs', exist_ok=True)

    df = pd.DataFrame({
        'Filename': filenames,
        'Predicted Label': Y_pred
    })

    # Sort numerically by filename
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else float('inf')

    df['FileNumber'] = df['Filename'].apply(extract_number)
    df = df.sort_values('FileNumber').drop(columns='FileNumber')

    # Label counts
    label_counts = df['Predicted Label'].value_counts().reset_index()
    label_counts.columns = ['Label', 'Count']

    output_path = f'../outputs/predictions_{dataset}.xlsx'
    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, sheet_name='Predictions', index=False)
        label_counts.to_excel(writer, sheet_name='Label Counts', index=False)

    print(f"\nPredictions written to Excel at: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['mnist', 'elegans'], default='mnist', help='Dataset to evaluate')
    args = parser.parse_args()

    dataset = args.dataset

    X_test, Y_test_labels, filenames, n_features, n_classes, model_path = load_test_images(dataset)
    model = load_model(model_path, n_features, n_classes)
    Y_pred_labels, accuracy, cm = evaluate(model, X_test, Y_test_labels)

    print(f"\n=== Evaluation Configuration ===")
    print(f"Dataset       : {dataset.upper()}")
    print(f"Model Path    : {model_path}\n")

    plot_confusion_matrix(cm, dataset)
    show_sample_predictions(X_test, Y_test_labels, Y_pred_labels, dataset)
    save_predictions_to_excel(filenames, Y_pred_labels, dataset)
