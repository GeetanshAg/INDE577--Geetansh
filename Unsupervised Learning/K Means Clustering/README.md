---

# Customer Segmentation Using K-Means Clustering

This Python project demonstrates the application of the **K-Means clustering algorithm** to segment customers based on their annual income and spending score. The dataset used is `Mall_Customers.csv`, which contains information about customers' annual income and spending scores.

## Algorithm Description: K-Means Clustering

K-Means clustering is an unsupervised machine learning algorithm that divides a dataset into **K** clusters, where each cluster is represented by its **centroid** (the mean of all the data points in that cluster). The algorithm works iteratively to assign data points to the nearest centroid and then re-calculates the centroids based on the mean of the assigned points. This process repeats until convergence, meaning that the centroids no longer change significantly.

### Key Steps in K-Means Algorithm:
1. **Initialization:** 
   - Select **K** initial centroids randomly from the dataset.
   
2. **Assignment:**
   - Assign each data point to the nearest centroid, forming K clusters.
   
3. **Update:**
   - Recalculate the centroids by computing the mean of all points assigned to each centroid.

4. **Repeat:**
   - Repeat the assignment and update steps until the centroids do not change or change very little, signaling that convergence has been reached.

### Advantages of K-Means:
- **Efficiency:** The algorithm is computationally efficient and scales well with large datasets.
- **Simplicity:** K-Means is easy to understand and implement.
- **Versatility:** It can be used for various clustering problems, including customer segmentation, image compression, and anomaly detection.

### Limitations of K-Means:
- **Choice of K:** The number of clusters (K) must be predefined, and the algorithm is sensitive to this choice.
- **Sensitive to Initial Centroids:** The results can vary depending on the initial placement of centroids, which may lead to suboptimal clustering.
- **Non-spherical Clusters:** K-Means tends to work best for spherical clusters and may not perform well when clusters have arbitrary shapes.

For this project, the **Elbow Method** is used to determine the optimal number of clusters.

## Code Explanation

1. **Data Loading and Preprocessing:**
   - The dataset is loaded into a Pandas DataFrame using `pd.read_csv()`.
   - We extract the relevant features (`Annual Income` and `Spending Score`) for clustering using `.iloc[]`.

2. **Elbow Method:**
   - The Within-Cluster Sum of Squares (WCSS) is computed for different values of K (from 1 to 10).
   - The Elbow Method graph is plotted to visualize the optimal number of clusters, which is determined by the "elbow" point in the graph.

3. **K-Means Clustering:**
   - K-Means is applied with **5 clusters** (based on the Elbow Method's output).
   - The cluster assignments are predicted using `kmeans.fit_predict()`.

4. **Visualization:**
   - The resulting clusters are visualized using `matplotlib`, with each cluster represented by a different color. The centroids of the clusters are also displayed.

```python
# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading the dataset and selecting specific columns
dataset = pd.read_csv('/content/Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Importing KMeans and applying the Elbow Method to determine the optimal number of clusters
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method graph to visualize the WCSS values
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying KMeans clustering with 5 clusters
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```

## Conclusion

The **K-Means clustering algorithm** effectively segments customers into distinct groups based on their annual income and spending score. This segmentation is valuable for businesses to create targeted marketing strategies or tailor their offerings to different customer segments.

While **K-Means** is efficient and widely used, it has limitations, such as the need to specify the number of clusters **K** in advance and its sensitivity to the initialization of centroids. For datasets with arbitrary cluster shapes, alternative clustering algorithms like **DBSCAN** or **Hierarchical Clustering** may be more appropriate.

