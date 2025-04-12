"""
[10] Implement a MeanShift Clustering of color images (either Java with ImageJ or Python).
- The user thereby enters the bandwidth as only input.
- Show the final clustered color image.
- Evaluate the final number of clusters on various color images.
- How do results depend on the provided bandwidth?

Implementation should contain the subsequent method definitions (given in Java-Style):
- Vector<double[]> MeanShiftColorPixel(Vector<double[]> inPixels, bandwidth) ==> one entire optimization run
- double ColourDist(double[] refColor, double[] currColor) ==> calculating distance between to positions in 3D RGB space
- double GaussianWeight(double dist, double bandwidth) ==> calculating the weight from neighbouring distance by using Gaussian kernel

[4] Visualization I: Animate the resulting color pixels in 3D color space AND 2D image space while performing the mean shift algorithm.

[4] Visualization II: Visualize the 3D color density topography.
"""
import numpy as np
from joblib import Parallel, delayed


def mean_shift_color_pixel(in_pixels: np.ndarray, bandwidth: float, epsilon: float = 1e-3, max_iter: int = 1000, iteration_callback=None):
    """
    Performs one complete Mean Shift optimization run on a set of pixel color values.

    This function iteratively shifts all input pixels in color space towards areas of higher density,
    using the Mean Shift algorithm with a Gaussian kernel. The shifting stops when all pixels have
    converged (i.e., their position changes are below the epsilon threshold), or the maximum number
    of iterations is reached. Optionally, a callback can be used to visualize intermediate results.

    :param in_pixels: Input image as a NumPy array of shape (H, W, 3) or flattened (N, 3),
                      where each pixel is represented by an RGB color vector.
    :param bandwidth: The radius of influence for shifting (i.e., the bandwidth)
    :param epsilon: Convergence threshold. If the shift for a pixel in one iteration is less than this
                    value, the pixel is considered to have converged.
    :param max_iter: Maximum number of iterations to perform before stopping the algorithm.
    :param iteration_callback: Optional function called after each iteration. Receives the current
                               shifted pixel array, iteration count, and original image shape.
    :returns: A tuple containing:
        - out_pixels: The final shifted image (same shape as in_pixels). Each pixel has been moved
                      in RGB color space to the mode of its local neighborhood.
        - clusters_original: A list of lists, where each inner list contains the original pixels that
                             belong to the same cluster.
        - cluster_centers: A list of cluster centers, computed as the centroid of all shifted points
                           assigned to that cluster.
    """
    original_pixels = in_pixels.copy()
    shifted_pixels = in_pixels.copy()
    if original_pixels.ndim == 3:
        original_pixels = original_pixels.reshape((-1, 3))
        shifted_pixels = shifted_pixels.reshape((-1, 3))

    converged = False
    iter_count = 0
    clusters_centered = []
    clusters_original = []
    still_shifting = [True] * shifted_pixels.shape[0]

    while not converged and iter_count < max_iter:
        curr_shifted_pixels = shifted_pixels.copy()
        results = Parallel(n_jobs=-1)(
            delayed(process_pixel)(i, shifted_pixels[i], still_shifting[i], curr_shifted_pixels, bandwidth, epsilon)
            for i in range(shifted_pixels.shape[0])
        )

        converged = True
        for j, new_p, still_moving, _ in results:
            shifted_pixels[j] = new_p
            still_shifting[j] = still_moving
            if still_moving:
                converged = False

        iter_count += 1
        if iteration_callback:
            step_data = MeanShiftStepData(shifted_pixels, iter_count, max_iter, in_pixels.shape, results)
            iteration_callback(step_data)

    if iter_count >= max_iter:
        print("Maximum number of iterations reached. Clusters cannot be generated.")
    else:
        for i, p in enumerate(shifted_pixels):
            add_point_to_clusters(clusters_centered, clusters_original, p, original_pixels[i], epsilon)

    out_pixels = shifted_pixels.reshape(in_pixels.shape)
    cluster_centers = get_centroids(clusters_centered)
    return out_pixels, clusters_original, cluster_centers


def process_pixel(i, p, active, original_pixels, bandwidth, epsilon):
    """
    Processes a single pixel in the Mean Shift optimization step.

    :param i: Index of the current pixel
    :param p: Current position of the pixel
    :param active: Boolean indicating whether this pixel is still moving
    :param original_pixels: All original pixel positions
    :param bandwidth: The bandwidth for the kernel
    :param epsilon: Convergence threshold
    :returns: Tuple (index, new pixel position, still active)
    """
    if not active:
        return (i, p, False, 0.0)

    new_p = mean_shift_step(p, original_pixels, bandwidth)
    distance = color_dist(new_p, p)

    if distance < epsilon:
        return (i, p, False, distance)
    else:
        return (i, new_p, True, distance)


def add_point_to_clusters(clusters_centered, clusters_original, clustered_point, original_point, epsilon: float):
    """
    Adds a point to the appropriate cluster or creates a new one if no match is found.

    :param clusters_centered: List of cluster centers (each a list of shifted points)
    :param clusters_original: List of corresponding original points for each cluster
    :param clustered_point: The shifted point to be clustered
    :param original_point: The corresponding original pixel
    :param epsilon: Distance threshold for considering points as part of the same cluster
    """
    for i, cluster_centered in enumerate(clusters_centered):
        for point in cluster_centered:
            dist = color_dist(clustered_point, point)
            if dist <= epsilon + 0.05:
                clusters_centered[i].append(clustered_point)
                clusters_original[i].append(original_point)
                return
    # If no suitable cluster was found, create a new cluster
    clusters_centered.append([clustered_point])
    clusters_original.append([original_point])


def get_centroids(clusters_centered):
    """
    Get the centroids for all clusters.

    :param clusters_centered: List of clusters where each cluster is a list of points
    :return: List of centroids, one for each cluster
    """
    centroids = [np.mean(cluster, axis=0) for cluster in clusters_centered]
    return centroids


def mean_shift_step(p: np.ndarray, points: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Performs a single mean shift step for a given point.

    :param p: The current pixel
    :param points: All original points
    :param bandwidth: The bandwidth for the kernel
    :returns: New shifted position as a NumPy array
    """
    shift = np.zeros(p.shape)
    total_weight = 0.0

    for p_temp in points:
        dist = color_dist(p, p_temp)
        weight = gaussian_weight(dist, bandwidth)
        shift += p_temp * weight
        total_weight += weight

    if total_weight == 0:
        return p

    return shift / total_weight


def color_dist(ref_color: np.ndarray, curr_color: np.ndarray) -> float:
    """
    Calculates the Euclidean distance between two RGB color vectors.

    :param ref_color: Reference color as a NumPy array of shape (3,)
    :param curr_color: Current color as a NumPy array of shape (3,)
    :returns: The Euclidean distance between the two colors
    """
    return float(np.linalg.norm(ref_color - curr_color))


def gaussian_weight(dist: float, bandwidth: float) -> float:
    """
    Calculates the weight using a Gaussian kernel based on distance and bandwidth.

    :param dist: Distance between color vectors
    :param bandwidth: Bandwidth parameter for the Gaussian function
    :returns: A weight value based on the Gaussian function
    """
    return np.exp(- (dist ** 2) / (2 * bandwidth ** 2)) # formula from the presentation
    #return (1 / (bandwidth * math.sqrt(2 * math.pi))) * np.exp(-0.5 * (dist / bandwidth) ** 2) # formula from wikipedia


class MeanShiftStepData:
    def __init__(self, shifted_pixels, current_iteration, max_iteration, original_image_shape, step_results):
        self.shifted_pixels = shifted_pixels
        self.current_iteration = current_iteration
        self.max_iteration = max_iteration
        self.original_image_shape = original_image_shape
        self.converged_pixel = sum(1 for _, _, still_moving, _ in step_results if not still_moving)
        self.total_pixels = self.shifted_pixels.shape[0]
        self.average_pixel_distance_from_centroid = sum(dist for _, _, _, dist in step_results) / self.total_pixels

    def print_step_results(self):
        print(f"Iteration {self.current_iteration} (Max Iteration: {self.max_iteration}, Image size: {self.original_image_shape[1]} x {self.original_image_shape[0]}):")
        print(f"    Number of converged pixels: {self.converged_pixel}/{self.total_pixels}")
        print(f"    Average distance of the pixels from their centroid: {self.average_pixel_distance_from_centroid:.4f}")
        print()