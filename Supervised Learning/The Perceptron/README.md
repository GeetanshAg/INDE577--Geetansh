# Single Layer Perceptron (SLP) for Binary Classification

## Description

This Python script demonstrates the implementation of a **Single Layer Perceptron (SLP)** algorithm for binary classification. The perceptron is a simple neural network model that learns to classify inputs into two classes based on a linear decision boundary.

### Artificial Neural Networks Overview

Artificial Neural Networks (ANNs) are computational models inspired by the human brain's structure and function. They are used to solve complex problems in areas such as machine learning and combinatorial optimization. ANNs adapt to data and improve over time without human intervention, making them powerful tools for tasks ranging from image recognition to predictive analytics.

### The Single Layer Perceptron (SLP)

The **Single Layer Perceptron** is one of the simplest forms of artificial neural networks. It is a linear classifier that works by using a set of weights, a bias term, and an activation function to classify inputs into one of two categories (binary classification). The perceptron’s learning algorithm was one of the first to demonstrate the possibility of machines learning from data and adjusting their parameters to make better predictions.

The perceptron works as follows:

1. It receives inputs, which are multiplied by associated weights.
2. The results are summed, and a bias term is added to the sum.
3. The sum is passed through an activation function, which determines the output.
4. The perceptron then classifies the input into one of the two classes based on whether the output is above or below a certain threshold.

The perceptron is trained using supervised learning, where it adjusts its weights and bias based on the errors made during the classification process.

## Key Components

### 1. Data Generation

The dataset is randomly generated using `numpy`, with two features (`x1` and `x2`). The target variable `y` is assigned based on whether the sum of `x1` and `x2` exceeds 1.

### 2. Model Training

The model is trained using the perceptron’s learning rule, where weights are updated iteratively to minimize the classification error. The training process involves repeatedly adjusting the weights after each classification attempt.

### 3. Visualization

The script includes functions to visualize the dataset and the decision boundary at each training iteration. The decision boundary is represented by the line where the perceptron classifies inputs as either 0 or 1.

## Algorithm Walkthrough

1. **Data Generation**: A dataset of random values is created for the input features `x1` and `x2`. The class labels `y` are generated based on a rule: `y = 1` if the sum of `x1` and `x2` is greater than 1, and `y = 0` otherwise.
   
2. **Training Process**: The perceptron is trained using the standard learning rule. At each step, the perceptron makes a prediction, compares it with the actual value, and adjusts its weights accordingly.

3. **Decision Boundary**: The decision boundary is plotted during the training process. This line separates the two classes and evolves as the perceptron learns to classify the data more accurately.

4. **Error Calculation**: The perceptron calculates the error after each training step and adjusts its weights to minimize the error.

## Functions

- **`show_dataset(data, ax)`**: Displays the dataset with class 1 (blue) and class 0 (red).
- **`testing(inputs)`**: Determines the class based on the sum of inputs.
- **`showAll(perceptron, data, threshold, ax)`**: Displays the dataset and decision boundary at a given threshold iteration.
- **`trainingData(SinglePerceptron, inputs)`**: Performs one iteration of training using the perceptron.
- **`limit(neuron, inputs)`**: Calculates the decision boundary (threshold line).
- **`show_threshold(SinglePerceptron, ax)`**: Plots the decision boundary on the graph.

## Image of Single Layer Perceptron

The following diagram explains the structure of the Single Layer Perceptron, illustrating the relationship between inputs, weights, bias, and the output:

![Single Layer Perceptron](sandbox:/mnt/data/A_diagram_explaining_the_Single_Layer_Perceptron_(.png)

## Training Visualization

The training process is visualized over 12 iterations, allowing you to see how the perceptron learns and adjusts the decision boundary after each training step.

## Conclusion

The Single Layer Perceptron is a simple yet powerful model for binary classification tasks. Although it is limited to linear classification, it serves as the foundation for more complex neural network models. By iteratively adjusting its weights and bias, the perceptron can learn to classify data based on patterns in the input features. Despite its simplicity, the perceptron laid the groundwork for the development of more advanced machine learning models.

### Comparison with Other Models

While the Single Layer Perceptron is a basic model, it has limitations:

- **Linear Boundaries**: The perceptron can only classify data that is linearly separable. More complex models like **Multilayer Perceptrons (MLPs)** or **Support Vector Machines (SVMs)** can classify non-linearly separable data.
- **Training Efficiency**: The perceptron algorithm can be inefficient for large datasets or complex problems. In contrast, modern algorithms like **Gradient Descent** and deep learning architectures are more scalable and efficient.

Overall, the perceptron is an excellent starting point for understanding machine learning concepts, particularly in the realm of neural networks.

