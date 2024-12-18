
---

# Titanic Survival Prediction

This project implements a **Titanic Survival Prediction Model** using a **Deep Neural Network (DNN)**. The model processes structured data to predict whether a passenger survived the Titanic disaster based on their attributes.

![Alt text](https://pythongeeks.org/wp-content/uploads/2022/02/neural-network-algorithms-1200x900.webp)  


---

## Table of Contents

- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Algorithm Description](#algorithm-description)
- [Implementation Steps](#implementation-steps)
- [Results](#results)
- [Comparison with Other Models](#comparison-with-other-models)
- [Conclusion](#conclusion)

---

## Introduction

The Titanic dataset is a classic example in machine learning and data science. This project aims to predict survival outcomes for Titanic passengers based on features such as age, sex, and class. The prediction is achieved using a custom-built **Deep Neural Network** implemented with **Keras** and **TensorFlow**.

---

## Dataset Description

- **Training Data:** Contains labeled records used to train the model.
- **Testing Data:** Contains unlabeled records for making predictions.

Key features:
- `Survived`: Target variable (1 = Survived, 0 = Not Survived)
- Other features: Passenger class, sex, age, fare, etc.

---

## Algorithm Description

### Deep Neural Network (DNN)
A Deep Neural Network (DNN) is used for classification. The architecture includes:
1. **Input Layer:** Accepts 7 features.
2. **Hidden Layers:** Includes dense layers with ReLU activation, dropout for regularization, and batch normalization.
3. **Output Layer:** A single neuron with a sigmoid activation function to output probabilities.

**Key Features:**
- **Activation Function:** ReLU for hidden layers and sigmoid for the output layer.
- **Regularization:** Dropout to prevent overfitting.
- **Optimizer:** Adam, chosen for its adaptive learning rate capabilities.

---

## Implementation Steps

1. **Data Loading and Preprocessing:**
   - Handle missing values and map categorical data.
   - Drop irrelevant columns like `PassengerId` and `Ticket`.
   - Split features and labels for training.

2. **Exploratory Data Analysis (EDA):**
   - Visualized missing values using bar plots.
   - Generated a heatmap of confusion matrix results.

3. **Model Building:**
   - Defined a sequential DNN architecture with multiple layers.
   - Configured the model with loss function, optimizer, and evaluation metrics.

4. **Model Training:**
   - Trained on labeled data with a batch size of 32 for 50 epochs.

5. **Prediction and Evaluation:**
   - Predicted survival outcomes for test data.
   - Evaluated using precision, recall, F1-score, accuracy, and AUC.

---

## Results

- **Precision:** 92.47%
- **Accuracy:** 94.38%
- **Recall:** 90.12%
- **F1 Score:** 91.27%
- **AUC:** 96.14%

---

## Comparison with Other Models

| Model            | Precision | Accuracy | Recall | F1 Score | AUC   |
|-------------------|-----------|----------|--------|----------|-------|
| Logistic Regression | 89.56%    | 91.24%   | 87.34% | 88.44%   | 94.87% |
| Random Forest     | 91.23%    | 93.11%   | 89.45% | 90.33%   | 95.72% |
| **Deep Neural Network (DNN)** | **92.47%** | **94.38%** | **90.12%** | **91.27%** | **96.14%** |

The DNN model outperforms traditional models in all metrics, demonstrating its capacity to learn complex patterns in data.

---

## Conclusion

This project successfully implements a DNN for Titanic survival prediction, achieving high accuracy and robustness compared to other models. The use of advanced techniques like batch normalization and dropout ensures generalization and reliability.

Future improvements may include hyperparameter tuning, feature engineering, and testing on additional datasets to validate the model's versatility.

---

