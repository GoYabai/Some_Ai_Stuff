import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier, KDTree

# ==========================================
# PART 1: DATA PREPARATION
# ==========================================
# Training data: 8 points with 2 features (e.g., coordinates X1, X2)
X_train = np.array([
    [2.0, 4.0], [1.0, 3.0], [2.5, 5.0], [3.0, 2.0],  # Points mostly bottom-left
    [6.0, 8.0], [7.0, 7.0], [8.0, 6.0], [9.0, 8.0]   # Points mostly top-right
])

# Labels for training data (0: Class A, 1: Class B)
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# The new point we want to classify
x_test = np.array([5.0, 5.0])

# Number of neighbors to consider
k = 3

print("X_train:\n", X_train)
print("y_train:", y_train)
print("New point x_test:", x_test)
print("-" * 40)


# ==========================================
# PART 2: MANUAL KNN IMPLEMENTATION
# ==========================================
# TODO 1: Calculate Euclidean distances from x_test to all points in X_train
# - Hint: Use np.linalg.norm(X_train - x_test, axis=1)
distances = np.linalg.norm(X_train - x_test, axis=1)

# TODO 2: Find the indices of the 'k' nearest neighbors
# - Hint: Use np.argsort(distances) to get sorted indices, then slice the first 'k' elements [:k]
nearest_indices = np.argsort(distances)[:k]

# TODO 3: Get the labels of these nearest neighbors and find the majority class
# - Hint 1: Get neighbor labels using y_train[nearest_indices]
# - Hint 2: Use np.bincount().argmax() OR Counter().most_common(1)[0][0] to find the most frequent label.
neighbor_labels = y_train[nearest_indices]
predicted_class = np.bincount(neighbor_labels).argmax()

print("Manual KNN - Distances:", distances)
print("Manual KNN - Nearest indices:", nearest_indices)
print("Manual KNN - Neighbor labels:", neighbor_labels)
print("Manual KNN - Predicted Class:", predicted_class)
print("-" * 40)


# ==========================================
# PART 3: USING SKLEARN KNEIGHBORSCLASSIFIER
# ==========================================
# TODO 4: Initialize the KNeighborsClassifier model with n_neighbors=3
knn_model = KNeighborsClassifier(n_neighbors=3)

# TODO 5: Train (fit) the model and predict the class for x_test
# - Hint: Fit requires 2D arrays. x_test is 1D, so use x_test.reshape(1, -1) when predicting.
# Write code to fit the model here:
knn_model.fit(X_train, y_train)

# Write code to predict here:
predicted_class_sklearn = knn_model.predict(x_test.reshape(1, -1))

print("Sklearn KNN - Predicted Class:", predicted_class_sklearn)
print("-" * 40)


# ==========================================
# PART 4: OPTIMIZING SEARCH WITH KD TREE
# ==========================================
# In this part, we use KDTree to query the nearest neighbors extremely fast 
# without calculating all pairwise distances.

# TODO 6: Build the KD Tree using the training data
# - Hint: Initialize KDTree with X_train
tree = KDTree(X_train, leaf_size=1)

# TODO 7: Query the tree to find the 'k' nearest neighbors for x_test
# - Hint: Use tree.query(x_test.reshape(1, -1), k=3)
# - Note: tree.query() returns 2 arrays: distances and indices.
tree_distances, tree_indices = tree.query(x_test.reshape(1, -1), k=3)

print("KD Tree - Distances:", tree_distances)
print("KD Tree - Nearest indices:", tree_indices)

# Check if the result matches our manual calculation
if np.array_equal(nearest_indices, tree_indices[0]):
    print("\nAwesome! KD Tree found the exact same neighbors as your manual method!")