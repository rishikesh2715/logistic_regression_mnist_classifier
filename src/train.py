from model import LogisticRegression
import utils
import argparse
import time 


def train_model(X_train, Y_train, X_val, Y_val, num_epochs=300, learning_rate=0.5, model_path='../models/model.pkl', plot_title=''):
    """
    Train the logistic regression model on a selected dataset.

    :param X_train: Training features
    :param Y_train: Training labels (one-hot)
    :param X_val: Validation features
    :param Y_val: Validation labels (one-hot)
    :param num_epochs: Number of training epochs
    :param learning_rate: Learning rate for SGD
    :param model_path: Path to save trained model
    :param plot_title: Title suffix for plots

    :return: Trained model and training stats
    """
    # Initialize model
    n_features = X_train.shape[1]
    n_classes = Y_train.shape[1]
    model = LogisticRegression(n_features, n_classes)

    # Training loop setup
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Transpose for training
    X_train = X_train.T
    Y_train = Y_train.T
    X_val = X_val.T
    Y_val = Y_val.T

    start_time = time.time()
    for epoch in range(num_epochs):
        # start time

        train_loss = model.train_step(X_train, Y_train, learning_rate)

        # Validation
        _, val_loss = model.propagation(X_val, Y_val)
        train_acc = model.get_accuracy(X_train, Y_train)
        val_acc = model.get_accuracy(X_val, Y_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%")
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    # Save and plot
    model.save_trained_model(model_path)
    utils.plot_results(train_losses, val_losses, train_accuracies, val_accuracies, title_suffix=plot_title)

    return model, train_losses, val_losses, train_accuracies, val_accuracies


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--dataset', type=str, choices=['mnist', 'elegans'], default='mnist', help='Dataset to train on')
   args = parser.parse_args()

   dataset = args.dataset

   # Load data
   if dataset == 'mnist':
      X_train, Y_train, X_val, Y_val = utils.load_mnist_train()
      num_epochs = 300
      learning_rate = 0.5
   else:
      X_train, Y_train, X_val, Y_val = utils.load_elegans_train()
      num_epochs = 1000
      learning_rate = 0.0005

   # Training configuration
   print(f"\n=== Training Configuration ===")
   print(f"Dataset       : {dataset.upper()}")
   print(f"Epochs        : {num_epochs}")
   print(f"Learning Rate : {learning_rate}")
   print(f"Model Path    : ../models/trained_model_{dataset}.pkl\n")

   # Train
   model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
      X_train, Y_train, X_val, Y_val,
      num_epochs=num_epochs,
      learning_rate=learning_rate,
      model_path=f'../models/trained_model_{dataset}.pkl',
      plot_title=dataset.upper()
   )




