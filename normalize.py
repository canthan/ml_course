import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])


def normalize(input):
    min = np.min(input)
    max = np.max(input)
    normalized = (input - min) / (max - min)

    return normalized


def set_threshold(input, threshold=0.5):

    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            input[i, j] = 0 if input[i, j] < threshold else 1

    return input


print(normalize(arr))

# normalized
print(set_threshold(normalize(arr)))
