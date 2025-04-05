import numpy as np
import random

def apply_kmeans(image, k=4, max_iterations=10):
    """
    Applies K-Means clustering to an image.

    :param image: Input image (NumPy array)
    :param k: Number of clusters
    :param max_iterations: Maximum number of iterations
    :return: Segmented image using the cluster colors
    """
    height, width, _ = image.shape

    # Randomly initialize clusters (choose k random pixels as starting points)
    clusters = np.array(random.choices(image.reshape(-1, 3), k=k), dtype=np.float64)

    for _ in range(max_iterations):
        clusters = update_clusters(image, clusters)

    # Create the segmented image based on the cluster assignments
    segmented_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            best_cluster_idx = get_best_cluster_idx(image[y, x], clusters)
            segmented_image[y, x] = clusters[best_cluster_idx]

    return segmented_image


def color_dist(ref_color, curr_color):
    """ Computes the Euclidean distance between two colors (in RGB). """
    return np.linalg.norm(ref_color - curr_color)


def get_best_cluster_idx(rgb, clusters):
    """ Returns the index of the closest cluster center for a given color. """
    distances = np.linalg.norm(clusters - rgb, axis=1)
    return np.argmin(distances)


def update_clusters(image, clusters):
    """
    Iterates over all pixels and updates the cluster centers based on the average
    of the assigned colors.

    :param image: NumPy array with shape (H, W, 3) for RGB image
    :param clusters: NumPy array with shape (k, 3) for cluster centers
    :return: Updated cluster centers
    """
    height, width, _ = image.shape
    num_clusters = clusters.shape[0]

    # Arrays to accumulate the sum of assigned pixels and count of pixels per cluster
    new_cluster_sum = np.zeros((num_clusters, 3), dtype=np.float64)
    cluster_counts = np.zeros(num_clusters, dtype=np.int32)

    # Go through all pixels and assign them to the nearest cluster
    for y in range(height):
        for x in range(width):
            pixel = image[y, x]
            best_cluster_idx = get_best_cluster_idx(pixel, clusters)
            new_cluster_sum[best_cluster_idx] += pixel
            cluster_counts[best_cluster_idx] += 1

    # Compute the new cluster centers
    for i in range(num_clusters):
        if cluster_counts[i] > 0:
            clusters[i] = new_cluster_sum[i] / cluster_counts[i]

    return clusters
