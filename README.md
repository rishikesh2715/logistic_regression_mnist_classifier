# Logistic Regression for MNIST Digit Classification

This project implements logistic regression using the softmax function to classify handwritten digits from the MNIST dataset. It is part of Project 4 for ECE 4332 / ECE 5370 -- Machine Learning by Dr. Hamid Sari-Saraf @ Texas Tech University.

## 🧠 Overview

We use a simple linear model trained with cross-entropy loss and softmax activation to classify digits (0–9) from 28×28 grayscale images.

## 🗂️ Directory Structure
.
├── data/                  # Contains the MNIST dataset in CSV format
│   ├── mnist_test.csv     # Test data for evaluating the model
│   └── mnist_train.csv    # Training data for training the model
├── main.py                # Main script to train the model
├── models/                # Directory to save trained model files
├── outputs/               # Stores model predictions, confusion matrix etc.
├── README.md              # Project overview and instructions
├── report/                # Contains reports and visualizations
└── src/                   # Source code for training, testing and utility functions



## ⚙️ Features

- Input: Flattened 28x28 images (784 features)
- Output: 10-class probability vector using softmax
- Optimizer: Gradient Descent / SGD
- Metrics: Accuracy, Confusion Matrix, Execution Time
- Output: Predictions saved in an Excel file

## 🧪 Usage

1. **Train the model** using `main.py` or your training script.
2. **Test the model** by passing a folder of `.tif` test images and a trained model file.
3. **Results** will be saved in an Excel file with filenames and predicted labels.


## 🧑‍💻 Author(s)

Rishikesh -- rishikesh3304@gmail.com
Samir Hossain -- samir.hossain@ttu.edu



