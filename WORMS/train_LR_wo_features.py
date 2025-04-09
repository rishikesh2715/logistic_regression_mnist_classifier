import os
import time
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# PARAMETERS (adjust these as needed)
DATA_DIR = os.path.join("Celegans_ModelGen")  # Folder with subfolders "0" and "1" (assumed inside the scripts folder)
TARGET_SIZE = (64, 64)      # Resize images to this size; set to None to keep original (101x101)
USE_PCA = True              # Whether to use PCA for dimensionality reduction
PCA_VARIANCE = 0.95         # Retain 95% of variance if using PCA
RANDOM_STATE = 42

def load_images(data_dir, target_size=None):
    X = []
    y = []
    filenames = []
    
    # Loop over class folders "0" and "1"
    for label in ['0', '1']:
        folder_path = os.path.join(data_dir, label)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist.")
            continue
        # List only .png files and wrap in tqdm for a progress bar
        files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
        for file in tqdm(files, desc=f"Loading images from folder {label}"):
            filepath = os.path.join(folder_path, file)
            try:
                img = Image.open(filepath).convert("L")  # Ensure image is in grayscale
                if target_size is not None:
                    img = img.resize(target_size)
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize pixels to [0,1]
                X.append(img_array.flatten())  # Flatten the 2D image to 1D vector
                y.append(int(label))
                filenames.append(file)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
    return np.array(X), np.array(y), np.array(filenames)

def main():
    # Step 1: Load Data
    print("Starting data loading...")
    X, y, filenames = load_images(DATA_DIR, target_size=TARGET_SIZE)
    print(f"Data loading complete. Loaded {len(X)} images.")

    # Step 2: Split Data into train, validation, and test sets (70/15/15 split)
    print("Splitting data into train, validation, and test sets...")
    X_train, X_temp, y_train, y_temp, fn_train, fn_temp = train_test_split(
        X, y, filenames, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test, fn_val, fn_test = train_test_split(
        X_temp, y_temp, fn_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")

    # Step 3: Feature Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling complete.")

    # Step 4: (Optional) Dimensionality Reduction via PCA
    if USE_PCA:
        print("Performing PCA for dimensionality reduction...")
        pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        print(f"PCA complete. Reduced dimensionality from {X_train.shape[1]} to {X_train_pca.shape[1]}.")
    else:
        X_train_pca, X_val_pca, X_test_pca = X_train_scaled, X_val_scaled, X_test_scaled

    # Step 5: Hyperparameter Tuning for Logistic Regression using GridSearchCV
    print("Starting hyperparameter tuning for Logistic Regression...")
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    logreg = LogisticRegression(
        penalty='l2', 
        max_iter=1000, 
        solver='saga', 
        random_state=RANDOM_STATE
    )
    grid = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    t_start = time.time()
    grid.fit(X_train_pca, y_train)
    train_time = time.time() - t_start
    best_logreg = grid.best_estimator_
    print(f"Hyperparameter tuning complete. Best parameters: {grid.best_params_}")
    print(f"Training time: {train_time:.2f} seconds")
    
    # Step 6: Evaluate on Validation Set (optional step for tuning)
    print("Evaluating model on validation set...")
    y_val_pred = best_logreg.predict(X_val_pca)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Step 7: Final Evaluation on Test Set
    print("Evaluating model on test set...")
    t_start = time.time()
    y_test_pred = best_logreg.predict(X_test_pca)
    test_time = time.time() - t_start

    cm = confusion_matrix(y_test, y_test_pred)
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    print("Test set evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"Testing time: {test_time:.2f} seconds")

    # Step 8: Output predictions to an Excel file
    print("Writing predictions to Excel file...")
    results_df = pd.DataFrame({
        "filename": fn_test,
        "predicted_label": y_test_pred
    })
    label_counts = results_df["predicted_label"].value_counts().rename_axis('label').reset_index(name='count')

    excel_filename = "output_celegans.xlsx"
    with pd.ExcelWriter(excel_filename) as writer:
        results_df.to_excel(writer, sheet_name="Predictions", index=False)
        label_counts.to_excel(writer, sheet_name="LabelCounts", index=False)
    print(f"Results written to {excel_filename}")

if __name__ == '__main__':
    main()
