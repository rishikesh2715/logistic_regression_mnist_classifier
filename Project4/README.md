# Logistic Regression for MNIST Digit Classification

This project implements logistic regression using the softmax function to classify handwritten digits from the MNIST dataset. It is part of Project 4 for ECE 4332 / ECE 5370 -- Machine Learning by Dr. Hamid Sari-Saraf @ Texas Tech University.

## Large Files in the Repository
The mnist_train.csv is a large file (∼107 MB). Enable git large file storage (LFS) after cloning this repository.

```bash
git lfs install
```

## Overview

Implementation of a simple linear model trained with cross-entropy loss and softmax activation to classify digits (0–9) from 28×28 grayscale images.

## Directory Structure

```bash
├── data/                  # Contains the MNIST dataset in CSV format
│   ├── mnist_test.csv     # Test data for evaluating the model
│   └── mnist_train.csv    # Training data for training the model
├── main.py                # Main script to train the model
├── models/                # Directory to save trained model files
├── outputs/               # Stores model predictions, confusion matrix etc.
├── README.md              # Project overview and instructions
├── report/                # Contains reports and visualizations
└── src/                   # Source code for training, testing and utility functions
    ├── data.py            # Data loading and preprocessing functions
    ├── model.py           # Model definition and training functions
    └── utils.py           # Utility functions for evaluation and visualization
```


## Project Workflow
```
           ┌──────────────────────┐
           │    mnist_train.csv   │
           └────────┬─────────────┘
                    │
            (read + preprocess)
                    ▼
           ┌──────────────────────┐
           │     utils.py         │  <- handles loading, preprocessing,
           └────────┬─────────────┘     one-hot encoding, splitting
                    │
       ┌────────────┴────────────┐
       ▼                         ▼
train_data                 val_data
train_label               val_label

       │                         │
       └────────────┬────────────┘
                    ▼
           ┌──────────────────────┐
           │     model.py         │  <- defines the logistic regression model
           └────────┬─────────────┘     (weights, softmax, forward, predict)
                    │
                    ▼
           ┌──────────────────────┐
           │     train.py         │  <- training loop:
           └────────┬─────────────┘     - forward pass
                    │                   - compute loss
                    ▼                   - backprop/SGD
           ┌──────────────────────┐     - log metrics
           │ Trained model (file) │
           └────────┬─────────────┘
                    │
         saved using utils or model method

                    ▼
           ┌──────────────────────┐
           │   evaluate.py        │  <- loads test data and model
           └────────┬─────────────┘     - predicts on test set
                    │                   - builds confusion matrix
                    ▼                   - exports results to Excel
           ┌──────────────────────┐
           │ predictions.xlsx     │
           └──────────────────────┘
```


## Specifications

- Input: Flattened 28x28 images (784 features)
- Output: 10-class probability vector using softmax
- Optimizer: Gradient Descent / SGD
- Metrics: Accuracy, Confusion Matrix, Execution Time
- Output: Predictions saved in an Excel file

## Installation

1. **Create a virtual environment** to hold all libraries associated with project. (Code below is best practice. **However**, _note this can be done without the virtual environment, but may result in conflicts._)
```bash
python -m venv C:\path\to\new\virtual\environment
```

##### OR

If the **virtualenv** library is already installed:
```bash
virtualenv env
```

2. **Install all required libraries** from the requirements.txt file in the root directory.
```bash
python install -r requirements.txt
```

##### OR

If **pip** is already installed you can run the following:
```bash
pip install -r requirements.txt
```

3. **YAY** now all libraries should be successfully installed for use of the project!

## Usage

### MNIST Training & Evaluation

1. **Enter** into the `Hossain_Rishikesh_Wes/` directory using the following command.
```bash
cd Hossain_Rishikesh_Wes
```

2. **Train the model** using `run_inference_gui.py` script. The trained model will be saved in the `models/` directory.
```bash
python run_inference_gui.py
```

3. **Results** will be saved in an Excel file with filename `predictions.xlsx` in the `outputs/` directory.
```bash
outputs/
└── predictions_mnist.xlsx
```

### Elegans Worms Training & Evaluation

1. **Enter** into the `Hossain_Rishikesh_Wes` by running the following command.
```bash
cd Hossain_Rishikesh_Wes
```

2. **Test** the Elegans model by running the following command inside the `Hossain_Rishikesh_Wes/` directory.
```bash
python run_inference_gui.py
```

3. **Results** will be saved in an Excel file with filename `predictions.xlsx` in the `outputs/` directory.
```bash
outputs/
└── elegans_predictions.xlsx
```

## Author(s)

[![Project Authors](https://contrib.rocks/image?repo=rishikesh2715/logistic_regression_mnist_classifier&max=300)](https://github.com/rishikesh2715/logistic_regression_mnist_classifier/graphs/contributors)




