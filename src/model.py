# import data handling from the utils.py file
import utils
import numpy as np

"""
model.py

TODO:

1. Define a LogisticRegression class:
   - Initialize weights and biases
   - Include forward pass with softmax activation

2. Implement softmax function:
   - Apply softmax to logits

3. Implement prediction function:
   - Returns class probabilities and predicted labels (argmax)

4. Add loss computation here:
   - Cross-entropy loss function

5. Add gradient computation (if not handled in train.py):
   - Compute gradients w.r.t weights and biases

6. Add some way to load and reuse model:
   - save_model(path)
   - load_model(path)
"""

# Class for logistic regression
class LogisticRegression:
   # Pulling and initializing the data from Utils.py
   def __init__(self):
      # Get the data from Utils.py
      self.X_train, self.Y_train, self.X_val, self.Y_val = utils.load_mnist_train()

      # Initializing the dimensions
      n_features = self.X_train.shape[1]
      n_classes = self.Y_train.shape[1]

   # Sigmoid function
   def sigmoid(self, z):
      s = 1.0 / (1.0 + np.exp(-z))
      return s
   
   # Initializing the Weights and bias
   def initialize(self, n_features, n_classes):
      weights = np.zeros((n_features, n_classes))
      bias = np.zeros(n_classes)

      return weights, bias
   
   def propogation(self, weights, bias, X_train, Y_train):
      n = X_train.shape[1]

      z = np.dot(weights.T, X_train)+bias
      A = self.sigmoid(z)
      E = -1.0/n*np.sum(Y_train*np.log(A)+(1.0-Y_train)*np.log(1.0-A))

      # Gradient of the loss with respect to weights and bias
      dw = 1.0/n*np.dot(X_train, (A-Y_train).T)
      db = 1.0/n*np.sum(A-Y_train)

      cost = np.squeeze(E)

      grads = [dw, db]
    
      return grads, cost
   
   # Optimizing our model with gradient descent Algorithm
   def DescentAlgorithm(self, weights, bias, X_train, Y_train, num, learning_rate):
      # Blah blah blah
      pass




   
