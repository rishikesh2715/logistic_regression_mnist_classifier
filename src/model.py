# import data handling from the utils.py file
import utils
import numpy as np
import pickle

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
"""
    The Logistic Regression class below is using a binary classification 10 times instead of a multi-class classification. In a way it is a simpler implementation
    because it sets harder boundaries between each number using the sigmoid function. It looks and compares the correct number vs the rest. 
    Instead of the relationships between the two. That is why we have a jump within our
    training accuracy.
"""
class LogisticRegression:
    def __init__(self, n_features, n_classes):
        # Initialize model parameters based on input dimensions
        self.weights, self.bias = self.initialize(n_features, n_classes)
        self.n_features = n_features
        self.n_classes = n_classes
        self.velocity_w = np.zeros_like((self.weights))
        self.velocity_b = np.zeros_like((self.bias))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
   
    # Initializing the Weights and bias
    def initialize(self, n_features, n_classes):
        # weights = np.zeros((n_features, n_classes))
        weights = np.random.randn(n_features, n_classes) * 0.01
        bias = np.zeros(n_classes)
        return weights, bias


    def propagation(self, X, Y, lambda_reg=0.01):
        n = X.shape[1]

        z = np.dot(self.weights.T, X) + self.bias.reshape(-1, 1)
        A = self.softmax(z)
        loss = -1.0 / n * np.sum(Y * np.log(A + 1e-8))

        # L2 regularization loss
        l2_loss = lambda_reg / (2 * n) * np.sum(self.weights ** 2)
        loss = loss + l2_loss

        dw = 1.0 / n * np.dot(X, (A - Y).T)
        db = 1.0 / n * np.sum(A - Y, axis=1)

        return {"dw": dw, "db": db}, np.squeeze(loss)

   
    def train_step(self, X, Y, learning_rate):
        grads, cost = self.propagation(X, Y)

        # Momentum update
        beta = 0.9
        self.velocity_w = beta * self.velocity_w - learning_rate * grads["dw"]
        self.velocity_b = beta * self.velocity_b - learning_rate * grads["db"]

        self.weights += self.velocity_w
        self.bias += self.velocity_b
        
        # Update parameters
        # self.weights = self.weights - learning_rate * grads["dw"]
        # self.bias = self.bias - learning_rate * grads["db"]
        
        return cost
   

    def predict(self, X):
        z = np.dot(self.weights.T, X) + self.bias.reshape(-1, 1)
        A = self.softmax(z)
        Y_prediction = np.argmax(A, axis=0)
        return Y_prediction



    def get_accuracy(self, X, Y):
        Y_pred = self.predict(X)
        Y_true = np.argmax(Y, axis=0)
        accuracy = np.mean(Y_pred == Y_true) * 100.0
        return accuracy
    

    # Add save/load methods
    def save_trained_model(self, path):
        # Making sure the path ends in .pkl
        if not path.endswith('.pkl'):
            path = path + '.pkl'

        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
    

    # Loading a model with a .pkl file type.
    def load_model(self, path):
        if not path.endswith('.pkl'):
            path = path + '.pkl'

        with open(path, 'rb') as f:
            params = pickle.load(f)
            self.__dict__.update(params)






   
