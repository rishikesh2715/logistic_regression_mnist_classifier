import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re
import time
from inference_scripts.model import LogisticRegression
from inference_scripts import utils

from sklearn.metrics import confusion_matrix


def detect_dataset(test_folder):
    sample_files = os.listdir(test_folder)
    for f in sample_files:
        if f.endswith('.tif'):
            return 'mnist'
        elif f.endswith('.png'):
            return 'elegans'
    raise ValueError("Unable to detect dataset from file extensions. Use .tif for MNIST, .png for Elegans.")


def load_test_data(test_folder, dataset):
    filenames = sorted(os.listdir(test_folder))
    images = []
    for fname in filenames:
        path = os.path.join(test_folder, fname)
        img = utils.load_and_preprocess_image(path, dataset)
        images.append(img.flatten())

    X_test = np.array(images).T
    return X_test, filenames


def load_and_run_model(model_path, X_test):
    # Load model
    with open(model_path, 'rb') as f:
        model = LogisticRegression(1, 1)  # Dummy init
        model.load_model(model_path)
    

    preds = model.predict(X_test)
    return preds


def save_predictions_to_excel(filenames, preds, output_path='predictions_output.xlsx'):
    df = pd.DataFrame({
        'Filename': filenames,
        'Predicted Label': preds
    })

    label_counts = df['Predicted Label'].value_counts().reset_index()
    label_counts.columns = ['Label', 'Count']

    # Append label counts at the end of predictions
    filler = pd.DataFrame([['', '']] * 2, columns=df.columns)
    summary = pd.DataFrame({
        'Filename': label_counts['Label'].astype(str),
        'Predicted Label': label_counts['Count'].astype(str)
    })

    final_df = pd.concat([df, filler, summary], ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_excel(output_path, index=False)
    print(f"Saved predictions with summary to: {output_path}")


def run_mnist_inference(test_path, model_path):
    # Load test data (images)
    X_test, filenames = load_test_data(test_path, 'mnist')

    # Load model
    model = LogisticRegression(1, 1)  # Dummy init
    model.load_model(model_path)

    # Run inference
    preds = model.predict(X_test)

    # Save predictions to Excel
    save_predictions_to_excel(filenames, preds, output_path=f'outputs/predictions_mnist.xlsx')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True, help='Path to folder with test images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pkl file')
    args = parser.parse_args()

    print("\n=== Inference Started ===")

    dataset = detect_dataset(args.test_path)
    print(f"Detected Dataset: {dataset.upper()}")

    start_time = time.time()
    X_test, filenames = load_test_data(args.test_path, dataset)
    preds = load_and_run_model(args.model_path, X_test)
    end_time = time.time()

    print(f"Inference Time: {end_time - start_time:.2f} seconds")
    save_predictions_to_excel(filenames, preds, f'outputs/predictions_{dataset}.xlsx')
    # plot_sample_predictions(X_test, preds, filenames, dataset)

    print("\n=== Inference Completed ===")
