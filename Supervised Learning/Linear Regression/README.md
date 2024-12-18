
# Linear Regression Model for Salary Prediction

## Overview
This project demonstrates the use of **Linear Regression** to predict a person's salary based on their years of experience. The dataset contains information about individuals' years of experience and their corresponding salaries. The model is trained on a subset of the data and tested to evaluate its performance.

## Algorithm Description

The **Linear Regression** algorithm is a simple yet powerful statistical technique used for predictive modeling. It assumes a linear relationship between the input variable (independent variable) and the target variable (dependent variable). In this case, the independent variable is the "Years of Experience," and the dependent variable is the "Salary."

### Key Steps in the Algorithm:
1. **Data Preprocessing**: The dataset is first loaded, and the independent variable (years of experience) and dependent variable (salary) are extracted.
2. **Splitting the Data**: The data is split into training and testing sets using `train_test_split` from scikit-learn. This helps to evaluate the model's performance on unseen data.
3. **Model Training**: A **Linear Regression** model is instantiated and trained on the training dataset. The `fit` method is used to learn the relationship between years of experience and salary.
4. **Prediction**: The trained model is used to predict salaries based on the test dataset.
5. **Visualization**: The training and testing results are visualized using scatter plots and the regression line.

### Linear Regression Model:
The core idea behind linear regression is to minimize the difference between the observed data points and the predicted values. This is achieved by finding the best-fitting straight line through the data points, defined by the equation:

\[
y = mx + b
\]

Where:
- \(y\) is the predicted salary.
- \(m\) is the slope of the line (coefficient).
- \(x\) is the years of experience.
- \(b\) is the intercept of the line.

## Code Explanation

### Step 1: Importing Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```
These libraries are used for numerical operations, data visualization, and data manipulation.

### Step 2: Loading and Preparing the Dataset
```python
dataset = pd.read_csv('/content/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```
The dataset is read from a CSV file, and the features (`X`) and target variable (`y`) are extracted.

### Step 3: Splitting the Data
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
```
The data is split into 2 sets: one for training the model and the other for testing its performance.

### Step 4: Training the Model
```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```
A Linear Regression model is instantiated and trained on the training data using the `fit()` method.

### Step 5: Making Predictions
```python
y_pred = regressor.predict(X_test)
```
The trained model is used to predict salaries based on the test set.

### Step 6: Visualizing the Results
#### Training Set Visualization
```python
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

#### Test Set Visualization
```python
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

### Visualizations:
1. **Training Set Visualization**: A scatter plot is generated for the training data points, and the regression line is plotted over it.
2. **Test Set Visualization**: A similar plot is created for the test data to compare the actual vs. predicted values.

## Results

The model's ability to predict salaries based on years of experience is demonstrated through the visualizations. The regression line shows the predicted trend, and the scatter plots depict the actual data points. 

## Linear Regression Algorithm Diagram

Below is a diagram illustrating the Linear Regression algorithm:

![Linear Regression Algorithm Diagram](https://upload.wikimedia.org/wikipedia/commons/3/3a/Linear_regression.svg)


## Conclusion
Linear regression provides a simple and interpretable model for predicting continuous values like salary. However, it assumes a linear relationship between the features and the target variable. If the relationship is non-linear, other models like polynomial regression or decision trees might perform better.

## Comparison with Other Models
- **Linear Regression** is suitable when the relationship between the variables is approximately linear. However, it may not perform well if the data is non-linear or contains significant outliers.
- **Polynomial Regression** can model non-linear relationships by adding higher-degree terms to the input features.
- **Decision Trees** and **Random Forests** can capture complex, non-linear patterns without the need for explicitly transforming the features.

