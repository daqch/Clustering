import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import numpy as np
from numpy.random import choice
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as hierarchy


d1 = pd.read_csv("dataset1.csv", header=None, names=[
                 "x1", "x2"], sep=" ").to_numpy()
d2 = pd.read_csv("dataset2.csv", header=None,
                 names=["x1", "x2", "x3"], sep=" ").to_numpy()


def random_kmeans(d, k):
    min_cost = math.inf
    final_labels = None
    final_u = None
    print("####### K MEANS UNIFORM RANDOM ########")
    for x in range(1):
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
    for x in range(1):
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


# for k in range(2, 14, 2):
#     print(k, " CENTERS")
#     labels, u, cost = plus_plus(d1, k)
#     print("cost++: ", cost)
#     labes, u, cost = random_kmeans(d1, k)
#     print("cost random: ", cost)
#     labels, u, cost = plus_plus(d2, k)
#     print("cost ++: ", cost)
#     labes, u, cost = random_kmeans(d2, k)
#     print("cost random: ", cost)
#     print("******************** DONE WITH ",
#           k, " CENTERS ********************")


costs_d1_random = np.array([[13556.68171183152, 5117.569860533506, 3517.0257989701286,
                             2600.6848038475387, 2185.3560342906344, 1812.217389827596], range(2, 14, 2)])
#plt.plot(costs_d1_random[1], costs_d1_random[0])
# plt.savefig("costs_d1_random.png")
costs_d1_plus = np.array([[13556.68171183152, 5117.569860533506, 3840.1517879313046,
                           2600.6410394489285, 2185.442531659962, 1821.658432401678], range(2, 14, 2)])
#plt.plot(costs_d1_plus[1], costs_d1_plus[0])

costs_d2_plus = np.array([[969423.3498365802, 616664.4245684871, 459490.10845103284,
                           362094.11203082866, 309540.8229187726, 272652.2346455147], range(2, 14, 2)])
costs_d2_random = np.array([[969423.3498365802, 616664.8272264968, 459501.7242408081,
                             362164.6560362114, 308282.43437951035, 272644.0460806322], range(2, 14, 2)])
# plt.plot(costs_d2_random[1], costs_d2_random[0])
# plt.plot(costs_d2_plus[1], costs_d2_plus[0])
# plt.legend(["uniform random", "kmeans++"])
# plt.yscale('linear')
# #plt.legend(["uniform random", "kmeans++"])
# plt.title("W(c) vs k")
# plt.xlabel("k")
# plt.ylabel("W(c)")
# plt.savefig("costs_d2_both.png")
# labels, u, cost = plus_plus(d1, 8)
# plt.scatter(d1[:, 1], d1[:, 0], c=labels)
# plt.title("kmeans++ with k = 8")
# plt.savefig("scatter_++_k=8.png")
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# labels, u, cost = random_kmeans(d2, 8)
# ax.scatter(d2[:, 2], d2[:, 1], d2[:, 0], c=labels)
# plt.show()
# plt.figure()
# dendogram = hierarchy.dendrogram(hierarchy.linkage(
#     d2, method="average"), truncate_mode="lastp")
# plt.savefig("dendogram_average.png")

model = AgglomerativeClustering(
    n_clusters=2, affinity='euclidean', linkage='average')
model.fit(d2)
labels = model.labels_
# plt.scatter(d1[:, 1], d1[:, 0], c=labels)
# plt.savefig("d1_average")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(d2[:, 2], d2[:, 1], d2[:, 0], c=labels)
plt.show()
