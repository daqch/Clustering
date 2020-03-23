import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin


X = pd.read_csv("dataset1.csv", header=None, names=[
                "x1", "x2"], sep=" ").to_numpy()

d2 = pd.read_csv("dataset2.csv", header=None,
                 names=["x1", "x2", "x3"], sep=" ").to_numpy()


def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)

        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels


for i in range(5):
    centers, labels = find_clusters(X, 20)
for i in range(20):
    find_clusters(d2, 20)

plt.scatter(X[:, 0], X[:, 1], c=labels,
            s=50, cmap='viridis')
plt.show()
