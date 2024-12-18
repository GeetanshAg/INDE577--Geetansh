

---

# Decision Tree Regression: Salary Prediction  

## Description  

This project demonstrates the implementation of a **Decision Tree Regression model** to predict salaries based on position levels. The decision tree is a non-parametric machine learning algorithm that can effectively capture non-linear relationships in data.  

The project includes:  
- Data preprocessing and visualization  
- Building and training a Decision Tree Regression model  
- Predicting salary for a specific position level  
- Visualizing the regression results  

---

## Algorithm: Decision Tree Regression  

### Overview  
Decision Tree Regression is a supervised machine learning algorithm that partitions the feature space into smaller, simpler regions by creating decision rules based on input features. The algorithm is particularly useful for modeling non-linear relationships.  

### Working of the Algorithm:  
1. **Data Partitioning**:  
   - The feature space is split into smaller regions based on thresholds that minimize the variance within each region.  
   - For each feature, the algorithm finds a split point that best separates the data points.  

2. **Leaf Nodes and Predictions**:  
   - Once the data is split, the mean (or median) of the target values in each region (leaf node) is used as the predicted value for that region.  
   - This process results in a step-function-like behavior for predictions.  

3. **Recursive Splitting**:  
   - The splitting process is repeated recursively, creating a tree structure where each split adds a decision node.  
   - The process stops when a stopping condition is met, such as a maximum tree depth or minimum number of samples per leaf.  

### Key Features of Decision Tree Regression:  
- **Captures Non-Linearity**: The algorithm can model complex, non-linear relationships between features and target variables.  
- **Feature Importance**: Decision trees inherently provide insights into the relative importance of features.  
- **Overfitting Risk**: Without proper tuning (e.g., limiting depth or minimum samples per leaf), the model can overfit to the training data.  

### Advantages:  
- No need for feature scaling (e.g., normalization or standardization).  
- Intuitive and interpretable.  
- Handles both numerical and categorical data.  

### Limitations:  
- Prone to overfitting when the tree grows too deep.  
- Sensitive to slight changes in data, which can lead to a different tree structure.  

In this project, the Decision Tree Regression algorithm is implemented using the `DecisionTreeRegressor` from `scikit-learn`.

---

## Features  

- **Data Loading and Preprocessing**  
- **Model Training**  
- **Prediction for Custom Inputs**  
- **Visualization of Results**

---

## Installation  

Install the required Python libraries using pip:  
```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## Usage  

1. Clone the repository or download the script.  
2. Ensure the dataset `Position_Salaries.csv` is in the same directory.  
3. Run the Python script:  
   ```bash
   python decision_tree_regression.py
   ```

### Input/Output  

- **Input**: Position levels and salaries from the dataset.  
- **Prediction Input**: A position level (e.g., `6.5`).  
- **Output**: Predicted salary and a visualization of the decision tree regression results.  

---

## Example  

### Dataset Sample (`Position_Salaries.csv`)  
| Position Level | Salary   |  
|----------------|----------|  
| 1              | 45000    |  
| 2              | 50000    |  

### Predicted Salary  
For position level `6.5`, the predicted salary is stored in `y_pred`.

### Visualization  
A plot is displayed with:  
- **Red dots**: Actual data points  
- **Blue curve**: Predicted regression curve  

---

## Dependencies  

- `numpy`: Efficient numerical computations  
- `pandas`: Data manipulation and analysis  
- `matplotlib`: Data visualization  
- `scikit-learn`: Decision Tree Regression model implementation  

---

