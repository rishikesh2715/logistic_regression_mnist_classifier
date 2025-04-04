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

    # Sigmoid function
    def sigmoid(self, z):
        s = 1.0 / (1.0 + np.exp(-z))
        return s
   
    # Initializing the Weights and bias
    def initialize(self, n_features, n_classes):
        weights = np.zeros((n_features, n_classes))
        bias = np.zeros(n_classes)
        return weights, bias
   
    def propagation(self, X, Y):
        # X shape (n_features, n_samples)
        # Y shape (n_classes, n_samples)
        n = X.shape[1]

        # Forward pass
        z = np.dot(self.weights.T, X) + self.bias.reshape(-1, 1)  # Reshape bias for broadcasting
        A = self.sigmoid(z)
        E = -1.0/n * np.sum(Y * np.log(A + 1e-8) + (1.0 - Y) * np.log(1.0 - A + 1e-8))

        # Gradient computation
        dw = 1.0/n * np.dot(X, (A - Y).T)
        db = 1.0/n * np.sum(A - Y, axis=1)  # Sum across samples

        cost = np.squeeze(E)
        grads = {"dw": dw, "db": db}
    
        return grads, cost
   
    def train_step(self, X, Y, learning_rate):
        grads, cost = self.propagation(X, Y)
        
        # Update parameters
        self.weights = self.weights - learning_rate * grads["dw"]
        self.bias = self.bias - learning_rate * grads["db"]
        
        return cost
   
    def predict(self, X):
        # X shape should be (n_features, n_samples)
        A = self.sigmoid(np.dot(self.weights.T, X) + self.bias.reshape(-1, 1))
        Y_prediction = (A > 0.5).astype(int)
        return Y_prediction

    def get_accuracy(self, X, Y):
        predictions = self.predict(X)
        # print("debuggin predictions: ", predictions)                 # Just here to help me debug and make sure my predictions are working correclty and to visualize them better
        accuracy = 100.0 - np.mean(np.abs(predictions - Y)) * 100.0
        # print("debugging accuracy: ", accuracy)                      # Just here to help me debug and make sure my accuracy is working correctly
        return accuracy
    
    # Uncomment the code below to just save the model base / untrained. 
    """
    def save_base_model(self, path):
        if not path.endswith('.pkl'):
            path = path + '.pkl'
            
        # Create a new instance with same dimensions but fresh initialization
        fresh_model = LogisticRegression(self.n_features, self.n_classes)
        
        with open(path, 'wb') as f:
            pickle.dump(fresh_model.__dict__, f)
    """

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

    # Creating the final form of the logisitic regression model. Setting the number of iterations and the learning rate for the gradient decent.
    def model(self, X_train, Y_train, X_test, Y_test, num = 1000, learning_rate = 0.05):
        costs = []                                                   # initializing the costs as an array to store all the values

        for i in range(num):
            cost = self.train_step(X_train, Y_train, learning_rate)  # based off the number of iterations we get the cost for each iteration calling the train_step function
            if i % 100 == 0:                                         # On the 100th iteration we append the cost to the costs array
                costs.append(cost)

        Y_prediction_train = self.predict(X_train)                   # Prediction for the training, by calling the predict function
        Y_prediction_test = self.predict(X_test)                     # Prediction for the testing, by calling the predict function

        train_accuracy = self.get_accuracy(X_train, Y_train)         # Return the training accuracy as a variable
        test_accuracy = self.get_accuracy(X_test, Y_test)            # Return the testing accuracy as a variable

        # Creating a dictionary to store the results of our model.
        r = {
            "Y_prediction_train": Y_prediction_train,
            "Y_prediction_test": Y_prediction_test,
            "w": self.weights,
            "b": self.bias,
            "learning_rate": learning_rate,
            "num": num}
        
        print("Accuracy for Training: ", train_accuracy)
        print("Accuracy for Testing: ", test_accuracy)

        # Return the results or r (which is a dictionary)
        return r





   
