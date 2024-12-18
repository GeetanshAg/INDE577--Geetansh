

---

# Logistic Regression for Social Network Ads Prediction

This Python project utilizes Logistic Regression to predict whether individuals will purchase a product based on their age and estimated salary. The dataset used is 'Social_Network_Ads.csv', which contains user information and their purchasing behavior.

## Table of Contents

- [Introduction](#introduction)
- [Algorithm Description](#algorithm-description)
- [Dataset](#dataset)
- [Implementation Steps](#implementation-steps)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

Logistic Regression is a statistical method used for binary classification tasks, predicting the probability of a binary outcome based on one or more predictor variables.

In this project, we apply Logistic Regression to predict whether a user will purchase a product based on their age and estimated salary.

## Algorithm Description

Logistic Regression is a type of regression analysis used for binary classification problems. Unlike linear regression, which is used to predict continuous values, Logistic Regression is used to predict discrete outcomes, typically a binary outcome (such as yes/no, 0/1). 

The algorithm works by estimating the probabilities using the logistic (sigmoid) function. The logistic function is an S-shaped curve that maps any input value (from -∞ to +∞) into a range between 0 and 1, making it perfect for binary classification tasks. 

The formula for the logistic function is:

\[
\text{Logistic function:} \ \hat{y} = \frac{1}{1 + e^{-z}}
\]

Where:
- \( \hat{y} \) is the predicted probability.
- \( e \) is the base of the natural logarithm.
- \( z \) is the linear combination of the input features, i.e., \( z = b + \sum_{i=1}^{n} x_i \cdot w_i \), where \( b \) is the bias term, \( x_i \) are the input features, and \( w_i \) are the weights.

### Key Characteristics of Logistic Regression:
- **Binary Classification**: It is mainly used for tasks where the output variable is binary (e.g., success/failure, 1/0).
- **Probability Estimation**: It estimates the probability of the occurrence of an event, based on the input features.
- **Decision Boundary**: The decision boundary is the threshold that separates the two classes. By default, the threshold is 0.5, meaning that if the predicted probability is greater than or equal to 0.5, the model predicts class 1, and if it is less than 0.5, the model predicts class 0.
- **Optimization**: Logistic Regression uses a method called Maximum Likelihood Estimation (MLE) to find the optimal parameters (weights) that maximize the likelihood of the observed data.

Logistic Regression is a simple yet powerful algorithm that is computationally efficient and works well for linearly separable data. However, it may not perform well when the relationship between features and the target is non-linear.

## Dataset

The 'Social_Network_Ads.csv' dataset contains the following columns:

- **Age**: Age of the user
- **EstimatedSalary**: Estimated annual salary of the user
- **Purchased**: Binary target variable indicating whether the user purchased the product (1) or not (0)

## Implementation Steps

1. **Import Libraries**:
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   import pandas as pd
   ```

2. **Load Dataset**:
   ```python
   dataset = pd.read_csv('Social_Network_Ads.csv')
   X = dataset.iloc[:, [2, 3]].values
   y = dataset.iloc[:, -1].values
   ```

3. **Split Data**:
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
   ```

4. **Feature Scaling**:
   ```python
   from sklearn.preprocessing import StandardScaler
   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test = sc.transform(X_test)
   ```

5. **Train Logistic Regression Model**:
   ```python
   from sklearn.linear_model import LogisticRegression
   classifier = LogisticRegression(random_state=0)
   classifier.fit(X_train, y_train)
   ```

6. **Make Predictions**:
   ```python
   y_pred = classifier.predict(X_test)
   ```

7. **Evaluate Model**:
   ```python
   from sklearn.metrics import confusion_matrix
   cm = confusion_matrix(y_test, y_pred)
   print(cm)
   ```

8. **Visualize Results**:
   ```python
   from matplotlib.colors import ListedColormap
   X_set, y_set = X_train, y_train
   X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                        np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
   plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha=0.75)
   plt.xlim(X1.min(), X1.max())
   plt.ylim(X2.min(), X2.max())
   for i, j in enumerate(np.unique(y_set)):
       plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1])
   plt.title('Logistic Regression (Training set)')
   plt.xlabel('Age')
   plt.ylabel('Estimated Salary')
   plt.legend()
   plt.show()
   ```

## Results

The confusion matrix provides the following results:

```
[[64  4]
 [ 3 29]]
```

This indicates that the model correctly predicted 64 non-purchases and 29 purchases, with 4 false positives and 3 false negatives.

## Conclusion

Logistic Regression effectively predicts user purchasing behavior based on age and estimated salary. The decision boundary visualization demonstrates how the model classifies users into purchase and non-purchase categories.

While Logistic Regression is a powerful tool for binary classification, it may not capture complex, non-linear relationships as effectively as other algorithms like Support Vector Machines (SVM) or Random Forests.

For more complex datasets, exploring these alternative models could yield better performance.

---

