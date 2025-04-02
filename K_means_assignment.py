import numpy as np
import matplotlib.pyplot as plt
from utils.k_means import *

# UNQ_C1
# GRADED FUNCTION: find_closest_centroids


def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example

    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): (K, n) centroids

    Returns:
        idx (array_like): (m,) closest centroids

    """

    # Set K
    K = centroids.shape[0]
    n = X.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    ### START CODE HERE ###
    for i in range(n):
        distance = []
        for j in range(K):

            j = np.linalg.norm([X[i] - centroids[j]])

            distance.append(j**2)

        min_distance = np.argmin(distance)

        idx[i] = min_distance

     ### END CODE HERE ###

    return idx


# Load an example dataset that we will be using
X = load_data()

print("First five elements of X are:\n", X[:5])
print('The shape of X is:', X.shape)

# Select an initial set of centroids (3 Centroids)
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find closest centroids using initial_centroids
idx = find_closest_centroids(X, initial_centroids)

# Print closest centroids for the first three elements
print("First three elements in idx are:", idx[:3])


# UNQ_C2
# GRADED FUNCTION: compute_centroids

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.

    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids

    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    lenghts = np.zeros(K, dtype=int)

    ### START CODE HERE ###
    for i in range(m):
        centroids[idx[i]] += X[i]
        lenghts[idx[i]] += 1

    for i in range(K):
        centroids[i] = centroids[i] / lenghts[i]

    ### END CODE HERE ##

    return centroids


K = 3
centroids = compute_centroids(X, idx, K)

print("The centroids are:", centroids)

# You do not need to implement anything for this part


def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):

        # Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))

        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)

        # Optionally plot progress
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids

        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show()
    return centroids, idx


# Load an example dataset
X = load_data()

# Set initial centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Number of iterations
max_iters = 10

# Run K-Means
centroids, idx = run_kMeans(
    X, initial_centroids, max_iters, plot_progress=True)
