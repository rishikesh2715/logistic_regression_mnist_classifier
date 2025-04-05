"""
train.py

TODO:

1. Load training data:
   - Use utils.py to load training/validation sets; it already preprocesses the data

2. Initialize logistic regression model:
   - From model.py

3. Implement training loop:
   - Forward pass (compute predictions)
   - Compute loss (cross-entropy)
   - Backward pass (compute gradients)
   - Update weights using SGD

4. Track performance:
   - Store training/validation loss
   - Track accuracy on both sets

5. Save training results:
   - Save model to disk
   - Save training logs (loss curves, accuracy, etc.)

6. Plot learning curves:
   - Use utils.plot_results()
"""

from model import LogisticRegression
import utils

def train_model(num_epochs=500, learning_rate=0.5):
    # Load data
   X_train, Y_train, X_val, Y_val = utils.load_mnist_train()

   # Initialize model
   n_features = X_train.shape[1]
   n_classes = Y_train.shape[1]
   model = LogisticRegression(n_features, n_classes)

   # Training loop
   train_losses = []
   val_losses = []
   train_accuracies = []
   val_accuracies = []

   # Transpose data once before training loop
   X_train = X_train.T  # Shape: (n_features, n_samples)
   Y_train = Y_train.T  # Shape: (n_classes, n_samples)
   X_val = X_val.T
   Y_val = Y_val.T
    
   for epoch in range(num_epochs):
      train_loss = model.train_step(X_train, Y_train, learning_rate)
         
      # Compute validation loss and accuracy
      _, val_loss = model.propagation(X_val, Y_val)
      train_acc = model.get_accuracy(X_train, Y_train)
      val_acc = model.get_accuracy(X_val, Y_val)
         
      """
      Logging the metrics every epoch. Logging every 100 epoch will miss
      a lot of valuable trend information when it comes to plotting
      the learning curves.
      """
      train_losses.append(train_loss)
      val_losses.append(val_loss)
      train_accuracies.append(train_acc)
      val_accuracies.append(val_acc)

      if epoch % 10 == 0:
         # We can print the metric every 10 epochs or 100. Too frequent will clutter the output
         print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
         print(f"Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%")
   
    # Saving the trained model
   model.save_trained_model('../models/trained_model.pkl')

   # Plotting the results
   utils.plot_results(train_losses, val_losses, train_accuracies, val_accuracies)
   
   return model, train_losses, val_losses, train_accuracies, val_accuracies

if __name__ == '__main__':
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_model()
