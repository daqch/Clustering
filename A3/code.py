import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import numpy as np
from numpy.random import choice


d1 = pd.read_csv("dataset1.csv", header=None, names=[
                 "x1", "x2"], sep=" ").to_numpy()
d2 = pd.read_csv("dataset2.csv", header=None,
                 names=["x1", "x2", "x3"], sep=" ").to_numpy()


def random_kmeans(d, k):
    min_cost = math.inf
    final_labels = None
    final_u = None
    print("####### K MEANS UNIFORM RANDOM ########")
    for x in range(5):
        clusters = []
        indexes = random.sample(range(0, len(d)), k)
        clusters = d[indexes]
        labels, u, cost = k_means(clusters, k, d)
        #print("Cost from iteration: ", x, " is: ", cost)
        if (cost < min_cost):
            final_labels = labels
            final_u = u
            min_cost = cost
    return final_labels, final_u, min_cost


def plus_plus(d, k):
    final_cost = math.inf
    final_u = final_labels = None
    print("####### K MEANS PLUS PLUS ##########")
    for x in range(5):
        clusters = np.array(d[random.randint(0, len(d))], ndmin=2)
        while len(clusters) < k:
            p = [None] * len(d)
            distances = [None] * len(d)
            # calculate distances to existing clusters
            for i in range(0, len(d)):
                # for each point
                distance = math.inf
                # calculate squared distance to closest center
                for j in range(len(clusters)):
                    if (np.linalg.norm(clusters[j]-d[i]) < distance):
                        distance = pow(np.linalg.norm(clusters[j]-d[i]), 2)
                        distances[i] = distance
            s = np.sum(distances)
            for i in range(len(d)):
                p[i] = distances[i] / s
            c = choice(range(len(d)), 1, p)
            clusters = np.concatenate((clusters, d[c]))
        labels, u, cost = k_means(clusters, k, d)
        #print("Cost from iteration: ", x, " is: ", cost)
        if (cost < final_cost):
            final_labels = labels
            final_cost = cost
            final_u = u
    return final_labels, final_u, final_cost


def k_means(u, k, d):
    cost = 0
    while True:
        labels = [None] * len(d)
        # Cluster points to means
        for x in range(0, len(d)):
            distance = math.inf
            for j in range(0, len(u)):
                if (np.linalg.norm(u[j]-d[x]) < distance):
                    distance = np.linalg.norm(u[j]-d[x])
                    labels[x] = j
        # Update the means with center points
        new_clusters = np.copy(u)
        n = [0] * len(new_clusters)
        for x in range(0, len(d)):
            if ((d[x] != u[labels[x]]).all()):
                new_clusters[labels[x]] += d[x]
            n[labels[x]] += 1
        for i in range(len(new_clusters)):
            new_clusters[i] = new_clusters[i]/n[i]
        # Check if they have change
        if((new_clusters == u).all()):
            # check cost
            for p in range(len(d)):
                cost += pow(np.linalg.norm(d[p] -
                                           u[labels[p]]), 2)
            break
        else:
            u = new_clusters
    return labels, u, cost


labels, u, cost = plus_plus(d1, 5)
print("cost++: ", cost)
labes, u, cost = random_kmeans(d1, 5)
print("cost random: ", cost)
labels, u, cost = plus_plus(d2, 5)
print("cost ++: ", cost)
labes, u, cost = random_kmeans(d2, 5)
print("cost random: ", cost)
# plt.scatter(d1[:, 0], d1[:, 1], c=labels)
# plt.show()
