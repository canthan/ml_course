import numpy as np
import matplotlib.pyplot as plt
from utils.tree_assignment import *


X_train = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1], [
                   0, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 0, 0]])
y_train = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])

print("First few elements of X_train:\n", X_train[:5])
print("Type of X_train:", type(X_train))

print("First few elements of y_train:", y_train[:5])
print("Type of y_train:", type(y_train))

print('The shape of X_train is:', X_train.shape)
print('The shape of y_train is: ', y_train.shape)
print('Number of training examples (m):', len(X_train))

# UNQ_C1
# GRADED FUNCTION: compute_entropy


def compute_entropy(y):
    """
    Computes the entropy for 

    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)

    Returns:
        entropy (float): Entropy at that node

    """
    # You need to return the following variables correctly
    entropy = 0.

    ### START CODE HERE ###
    print(y)
    if len(y) == 0:
        return 0

    p1 = 0.
    n = y.shape[0]

    for i in range(n):
        if y[i] == 1:
            p1 += 1

    p1 = p1 / n

    if p1 == 0 or p1 == 1:
        entropy = 0

    else:
        entropy = -p1 * np.log2(p1) - ((1 - p1) * np.log2(1 - p1))
        print(entropy)

    ### END CODE HERE ###

    return entropy


print("Entropy at root node: ", compute_entropy(y_train))


# UNQ_C2
# GRADED FUNCTION: split_dataset

def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches

    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on

    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
    """

    # You need to return the following variables correctly
    left_indices = []
    right_indices = []

    ### START CODE HERE ###
    for i in node_indices:
        if X[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)

    ### END CODE HERE ###

    return left_indices, right_indices

# Case 1


root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Feel free to play around with these variables
# The dataset only has three features, so this value can be 0 (Brown Cap), 1 (Tapering Stalk Shape) or 2 (Solitary)
feature = 0

left_indices, right_indices = split_dataset(X_train, root_indices, feature)

print("CASE 1:")
print("Left indices: ", left_indices)
print("Right indices: ", right_indices)

# Visualize the split
generate_split_viz(root_indices, left_indices, right_indices, feature)

print()

# Case 2

root_indices_subset = [0, 2, 4, 6, 8]
left_indices, right_indices = split_dataset(
    X_train, root_indices_subset, feature)

print("CASE 2:")
print("Left indices: ", left_indices)
print("Right indices: ", right_indices)

# Visualize the split
generate_split_viz(root_indices_subset, left_indices, right_indices, feature)


# UNQ_C3
# GRADED FUNCTION: compute_information_gain

def compute_information_gain(X, y, node_indices, feature):
    """
    Compute the information of splitting the node on a given feature

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        feature (int):           Index of feature to split on

    Returns:
        cost (float):        Cost computed

    """
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)

    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    # You need to return the following variables correctly
    information_gain = 0

    ### START CODE HERE ###
    entropy_n = compute_entropy(y_node)
    entropy_l = compute_entropy(y_left)
    entropy_r = compute_entropy(y_right)

    information_gain = entropy_n - \
        ((len(X_left) / len(X_node) * entropy_l) +
         (len(X_right) / len(X_node) * entropy_r))

    ### END CODE HERE ###

    return information_gain


info_gain0 = compute_information_gain(
    X_train, y_train, root_indices, feature=0)
print("Information Gain from splitting the root on brown cap: ", info_gain0)

info_gain1 = compute_information_gain(
    X_train, y_train, root_indices, feature=1)
print("Information Gain from splitting the root on tapering stalk shape: ", info_gain1)

info_gain2 = compute_information_gain(
    X_train, y_train, root_indices, feature=2)
print("Information Gain from splitting the root on solitary: ", info_gain2)


# UNQ_C4
# GRADED FUNCTION: get_best_split

def get_best_split(X, y, node_indices):
    """
    Returns the optimal feature and threshold value
    to split the node data 

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """

    # Some useful variables
    num_features = X.shape[1]

    # You need to return the following variables correctly
    best_feature = -1
    i_gain = 0

    ### START CODE HERE ###
    for i in range(num_features):
        i_gain_temp = compute_information_gain(X, y, node_indices, i)

        if i_gain_temp > i_gain:
            i_gain = i_gain_temp
            best_feature = i

    ### END CODE HERE ##

    return best_feature


best_feature = get_best_split(X_train, y_train, root_indices)
print("Best feature to split on: %d" % best_feature)
