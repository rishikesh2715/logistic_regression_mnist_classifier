from . import utils
import numpy as np
import pickle

# Class for logistic regression
"""
    The Logistic Regression class below is using a binary classification 10 times instead of a multi-class classification. In a way it is a simpler implementation
    because it sets harder boundaries between each number using the sigmoid function. It looks and compares the correct number vs the rest. 
    Instead of the relationships between the two. That is why we have a jump within our
    training accuracy.
"""
class LogisticRegression:
    def __init__(self, n_features, n_classes):
        """
        Initialize the Logistic Regression model.
        :param n_features: Number of features in the input data
        :param n_classes: Number of classes in the output data
        """
        # Initialize model parameters based on input dimensions
        self.weights, self.bias = self.initialize(n_features, n_classes)
        self.n_features = n_features
        self.n_classes = n_classes
        self.velocity_w = np.zeros_like((self.weights))
        self.velocity_b = np.zeros_like((self.bias))


    def softmax(self, z):
        """
        Compute the softmax of the input array z.
        :param z: Input array of shape (n_classes, n_samples)
        :return: Softmax probabilities of shape (n_classes, n_samples)
        """
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

   
    def initialize(self, n_features, n_classes):
        """
        Initialize weights and bias for the model.
        :param n_features: Number of features in the input data
        :param n_classes: Number of classes in the output data
        :return: Initialized weights and bias
        """
        weights = np.random.randn(n_features, n_classes) * 0.01
        bias = np.zeros(n_classes)
        return weights, bias


    def propagation(self, X, Y, lambda_reg=0.01):
        """
        Perform forward propagation and compute the loss.
        :param X: Input data of shape (n_features, n_samples)
        :param Y: One-hot encoded labels of shape (n_classes, n_samples)
        :param lambda_reg: Regularization parameter
        :return: Gradients and loss
        """
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
        """
        Perform a single training step.
        :param X: Input data of shape (n_features, n_samples)
        :param Y: One-hot encoded labels of shape (n_classes, n_samples)
        :param learning_rate: Learning rate for weight updates
        :return: Cost after the training step
        """
        # Compute gradients and cost
        grads, cost = self.propagation(X, Y)

        # Momentum update
        beta = 0.9
        self.velocity_w = beta * self.velocity_w - learning_rate * grads["dw"]
        self.velocity_b = beta * self.velocity_b - learning_rate * grads["db"]

        self.weights += self.velocity_w
        self.bias += self.velocity_b
      
        return cost
   

    def predict(self, X):
        """
        Predict the class labels for the input data.
        :param X: Input data of shape (n_features, n_samples)
        :return: Predicted class labels of shape (n_samples,)
        """
        z = np.dot(self.weights.T, X) + self.bias.reshape(-1, 1)
        A = self.softmax(z)
        Y_prediction = np.argmax(A, axis=0)
        return Y_prediction


    def get_accuracy(self, X, Y):
        """
        Compute the accuracy of the model on the input data.
        :param X: Input data of shape (n_features, n_samples)
        :param Y: One-hot encoded labels of shape (n_classes, n_samples)
        :return: Accuracy as a percentage
        """
        Y_pred = self.predict(X)
        Y_true = np.argmax(Y, axis=0)
        accuracy = np.mean(Y_pred == Y_true) * 100.0
        return accuracy
    

    def save_trained_model(self, path):
        """
        Save the trained model parameters to a file.
        :param path: Path to save the model file
        """
        # Making sure the path ends in .pkl
        if not path.endswith('.pkl'):
            path = path + '.pkl'

        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
    

    def load_model(self, path):
        """
        Load the model parameters from a file.
        :param path: Path to the model file
        """
        if not path.endswith('.pkl'):
            path = path + '.pkl'

        with open(path, 'rb') as f:
            params = pickle.load(f)
            self.__dict__.update(params)






   
