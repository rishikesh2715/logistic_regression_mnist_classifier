# Explanation of Model.py

## Logistic Regression Model

For our model, and given project we are implementing **Logistic Regression**.
Logistic Regression is a statistical model that models the log-odds of an event as a linear combination of one or more independent variables.
For Logistic Regression there are 2 types. The first is **binary classification** and the second being **multiclass classification**.

## Our Approach

Within Model.py we first created a class to store all the functions that would create our model.
Thus we named the class ```logisticRegression```. For the datasets, we decided to go with the **Multiclass classification**,
for one main reason. When using multiclass classification you can implement binary, 
so for best optimization it uses the **softmax function**:

### Step 1 - Initialization

When creating Logistic Regression, we need to initialize/define the parameters based on the input dimensions.
We define the input dimensions off of 2 different types. First is the number of features in the input data: ```n_features```
The second is the number of classes in the output data: ```n_classes```.

_**Note** that for Elegans the number of classes is 2: 1 = Worm. 0 = No worm. For the MNIST we will have 10 classes,
because we are identifying numbers between 0-9._

For initializing we will create an init function defined below:
```
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
```

### Step 2 - Implementing Softmax Function

For our second step, we are going to define the softmax equation below into code. The Softmax function output's a vector
of propabilities, where each element within the vector represents the probability of a certain class happening. Where the total value is summed to 1. It is an extension of the sigmoid function. The softmax function ensures that the output probabilities are between 0 and 1,
where they sum to 1, representing a valid probability distribution.


#### Online version of Softmax function:
```math
σ(\vec{z})_{i} = \frac{e^{z_{i}}}{\sum_{j=1}^{K} e^{z_{j}}}
```

#### In-class version of Softmax function:
```math
y^{(n)}_{j} = \frac{e^{\underline{w}_j \cdot{\phi{\underline{x}^{n}}}}}{\sum_{k=1}^{K} e^{\underline{w}_k}\cdot{\phi{\underline{x}^{n}}}}
```

#### Code Implementation of **Softmax Function**:

For the code, we create a function, where we pass itself, alongside the shape of the input array. (dimension).
We then return the Softmax probabilities of the shape.

```
def softmax(self, z):
        """
        Compute the softmax of the input array z.
        :param z: Input array of shape (n_classes, n_samples)
        :return: Softmax probabilities of shape (n_classes, n_samples)
        """
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
```

### Step 3 - Initialize the Weights & Bias

For Step 3, we need to define the Weights and Bias for the model. The **weights** within a Logistic Regression model determine the influence of each input feature on the prediction. While the **bias** shifts the boundary decision between each of the classes.

```
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
```


### Step 4 - Propagation

For step 4 we need to implement propagation which is where to train our model, we need the model to predict
the probability of an input belonging to a certain classification. Then when it selects the class with the highest probability as
the prediction.

```
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
```

### Step 5 - Step Training function

For the model to learn, we perform a "training step" where the model learns gradually over time. For our model,
we will implement a variant of Gradient Descent called "Gradient Descent with Momentum". It is similiar to normal gradient descent;
however, it incorporates the past update direction, which helps accelerate convergrnece and smoot out oscillations in the optimization path. This helps alot with noisy gradients in regards to the cost function.

```X``` is the input data with shape ```(n_features, n_samples)```, where each column represents a single data sample.
```Y``` is the One-hot encoded true labels with shape ```(n_features, n_samples)```, which corresponds to each column of X.

The **Learning rate** is used to update the model paramets at a certain step size. _defined as a float value_.
The Function will return the **cost** which is the computed cost (loss) after the weight and bias updates.

This method performs the training step by doing the following:
- Computing the gradients and cost using the `propagation` method.
- Updating the velocity terms for weights and biases using momentum.
- Updating the model's weights and biases using the velocity terms.

**Code Below:**
```
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
```

### Step 6 - Prediction

For step 6, we will define a function called `predict` where we have our model,
predicting the class labels for a given set of input samples using the trained logistic regression model.

For the function we will define 2 parameters, one being ```self``` which allows us to call this function within the given class. 
The second parameter we will define is `X` which is the input data of shape `(n_features, n_samples)`.

This method performs forward propagation to compute the model's output probabilities for each class using the
softmax activation function. It then selects the class with the highest probability as the predicted label for each input sample.

The method performs the above information by doing the following:
- Computes the linear logits: Z = W.T * X + b    _**Note** that W.T stands for the weights transposed since it is a vector._
- Applies the softmax activation to obtain class probabilities.
- Use `np.argmax` to selct thge most probable class for each sample.

##### Y Equation - (Probability of being within a given classification)

```math
y^{(n)}_{j} = P(C_{j} | \phi{(\underline{x}^{(n)})})
```

**Code Below:**
```
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
```

### Step 7 - Get the Accuracy

For step 7, we are defining a function that will compute the classification accuracy of the logistic regression model.
(How accurate based on a percentage is our model in identification).

For the function we will be passing in 3 parameters. `(self, X, Y)`. These 3 parameters do different things. Review previous instructions for what they do. Then our function **returns** the accuracy as a **float** value. Where the classication accuracy is written as a percentage.

This method evaluates the model's performance by doing the following:
- Predicting the class labels for input samples using `self.predict(X)`.
- Converting the one-hot encoded labels `Y` into class indices.
- Comparing the predicted labels against the true labels.
- Computing the mean accuracy and returning it as a percentage.

**Code Below:**
```
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
```

### Step 8 - Save the trained model

For step 8, we will save the trained model parameters to a file.
We will need to import a python library for this step for easier implementation.
```pip install pickle``` Then `import pickle` at the top of your code.
We will define 2 parameters for the function `self` and `path` where **path** is the file location.

**Code Below:**
```
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
```

### Step 9 - Load a saved model

For step 9, we will basically repeat step 8, however instead of writing the file to a certain path, we will be reading the file
from a certain path.

**Code Below:**
```
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
```