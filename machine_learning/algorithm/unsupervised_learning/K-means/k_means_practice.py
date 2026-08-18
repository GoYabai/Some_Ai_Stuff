import os

os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# ==========================================
# PART 1: DATA PREPARATION
# ==========================================
# Data includes age and expenditure[cite: 1]
age = np.array([18, 20, 22, 30, 34, 40, 60, 66, 70])
expenditure = np.array([80, 90, 85, 50, 64, 60, 30, 40, 25])

# Stack data into a matrix X of shape (9, 2)[cite: 1]
X = np.column_stack((age, expenditure))
print("Data X:\n", X)
print("-" * 30)


# ==========================================
# PART 2: MANUAL K-MEANS IMPLEMENTATION
# ==========================================
k = 3
max_iters = 2

# Initialize starting centroids (take points from index 2 to k+1 of X)[cite: 1]
centroids = X[2:k+2]
print('Init centroids: \n', centroids)

for _ in range(max_iters):
    
    # TODO 1: Assign points to clusters
    # - Hint: Use np.linalg.norm to calculate distances from X to centroids.
    # - Hint: Use np.argmin to find the labels for each point.
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    
    # TODO 2: Update centroids
    # - Hint: Use list comprehension looping i in range(k).
    # - Hint: Calculate the mean along axis 0 for points belonging to cluster i.
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    
    # Check for convergence[cite: 1]
    if np.all(centroids == new_centroids):
        break
        
    centroids = new_centroids
    
    # Print results after each iteration[cite: 1]
    print("Labels: \n", labels)
    print("Centroids:\n", centroids)
    print()

# TODO 3: Calculate WCSS (Within-Cluster Sum of Squares)
# - Hint: Calculate the sum of squared distances from each point in a cluster to its centroid.
wcss = np.sum([np.square(X[labels == i] - centroids[i]) for i in range(k)])
print("Manual WCSS:", wcss)
print("-" * 30)


# ==========================================
# PART 3: USING SKLEARN LIBRARY
# ==========================================
# TODO 4: Initialize and train the model using sklearn
# - Hint: Initialize KMeans with n_clusters=3[cite: 1].
# - Hint: Fit the data X (might need to use X.reshape(-1, 2))[cite: 1].
# - Hint: Extract labels_ and inertia_ from the model[cite: 1].
kmeans = KMeans(n_clusters=3)
# Write fit code here
kmeans.fit(X)
labels_sklearn = kmeans.labels_
wcss_sklearn = kmeans.inertia_

print("Labels from Sklearn:\n", labels_sklearn)
print("WCSS from Sklearn:", wcss_sklearn)

# Print result for each cluster[cite: 1]
# for x, label in zip(X, labels_sklearn):
#     print(f"Cluster {label}: {x}")
print("-" * 30)


# ==========================================
# PART 4: PLOTTING THE ELBOW METHOD TO FIND OPTIMAL K
# ==========================================
wcss_values = []

# Loop from 1 to 8 to try different values of k[cite: 1]
for i in range(1, 9):
    # TODO 5: Run K-Means for each value i and append WCSS to the wcss_values array
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    wcss = kmeans.inertia_
    wcss_values.append(wcss)

# Plot the graph to show results[cite: 1]
plt.plot(range(1, 9), wcss_values)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('WCSS vs. Number of Clusters')
plt.show()


# from the plot we can see that K = 3 is the optimal for this task