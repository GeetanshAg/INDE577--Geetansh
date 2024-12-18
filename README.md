# Machine Learning and Data Science
This repository comprises a compilation of machine learning algorithms that record the progress of a standard machine learning project. Machine learning can be classified into two primary categories: supervised and unsupervised. Supervised learning can be categorized into regression and classification techniques. Unsupervised learning can be separated into clustering and anomaly detection.

This repository follows the standard progression of a machine learning project: data organization and analysis, model selection, and model comparison. The selection of models is initially categorized into supervised and unsupervised, followed by regression and classification. The sole unsupervised learning technique we shall examine is clustering. Each approach uses the same dataset to make comparisons more tractable and to assist in highlighting the differences between the algorithms.


**Supervised versus Unsupervised**

Supervised learning involves training an algorithm using predefined labels and attributes. A label represents the prediction of interest, and its specific shape and meaning may vary between projects. For example, the label for a computer vision project normally employs a one-hot encoding approach the size of the number of objects the algorithm needs to learn while the label of a regression algorithm is a prediction created by fitting a line to data.

Unsupervised learning is employed to discern patterns and relationships within unlabeled data. For example, if you have a large dataset and would like to know how many groups most accurately represent the data you would use an unsupervised learning algorithm to calculate the number of clusters.


**Regression versus Classification**

Regression identifies the optimum line to minimize some cost function. This line can then be used to predict a continuous label based on input features. The most popular example of a regression method is a real estate dataset which uses multiple input features - such as number of bathrooms, number of bedrooms, house size, and lot size - to forecast the sale price of a home.

Classification employs input vectors to derive a 0/1 label anticipating if a combination of input vectors corresponds to a category or not. If there are numerous categories, this can be accomplished by utilizing one-hot encoding where there are one output for each potential category in which only one output can be 1 at a time.


**Clustering vs Anomaly Detection**

Clustering organizes the dataset into a specific number of clusters with the objective of establishing how the data is segregated or what groups future data will belong to. Clustering can be used for activities like market segmentation, search engines, and data analysis. In clustering, the number and characteristics of the labels are unknown and the programmer just specifies the number of probable clusters.

Anomaly detection constructs a statistical model of what "normal" data looks like and uses it to identify abnormal instances. This helps to spot damaged items or new trends without stating exactly what changes to look for.


**Resources**

In this repository, we will experience how each algorithm analyzes and trains on the data. This will be performed via a mix of sklearn approaches and self-programmed examples. This will present the reader with 
the ability to understand how to implement ready built ways to start their own machine learning project rapidly, as well as how to construct their own custom designed modules.
