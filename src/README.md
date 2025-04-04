# Explanation of Model.py

## Logistic Regression Model
For our model, and given project we are implementing Logistic Regression.
Logistic Regression is a statistical model that models the log-odds of an event as a linear combination of one or more independent variables.
For Logistic Regression there are 2 types. The first is binary classification and the second being multiclass classification.

## Our Approach
Within Model.py we first created a class to store all the functions that would create our model.
Thus we named the class ```logisticRegression```. For the datasets, we decided to go with the Binary classification for one main reason.
When using multiclass classification you can implement binary, but for best optimization it uses the **softmax function**:

#### Online version of Softmax function:
$$
σ(\vec{z})_{i} = \frac{e^{z_{i}}}{\sum_{j=1}^{K} e^{z_{j}}}
$$

#### In-class version of Softmax function:
$$y^{(n)}_{j} = \frac{e^{\underline{w}_j \cdot{\phi{\underline{X}^{n}}}}}{\sum_{k=1}^{K} e^{\underline{w}_k}}$$
