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

"""
for p in copied_points: while !convergence: p = mean_shift(p, original_points)
"""
def mean_shift_color_pixel(in_pixels: np.ndarray, bandwidth: float, epsilon: float = 1e-3, max_iter: int = 100, iteration_callback=None):
    """
    Performs one complete Mean Shift optimization run.

    :param in_pixels: List of color vectors (each a NumPy array of shape (3,))
    :param bandwidth: The radius of influence for shifting (i.e., the bandwidth)
    :param epsilon:
    :param max_iter:
    :param iteration_callback:
    :returns: A list of shifted pixel color vectors (potential cluster centers)
    """
    original_pixels = in_pixels.copy()
    shifted_pixels = in_pixels.copy()
    if original_pixels.ndim == 3:
        original_pixels = original_pixels.reshape((-1, 3))
        shifted_pixels = shifted_pixels.reshape((-1, 3))

    converged = False
    iter_count = 0
    clusters_original = []
    clusters_centered = []
    still_shifting = [True] * shifted_pixels.shape[0]
    while not converged and iter_count < max_iter:
        converged = True
        for i, p in enumerate(shifted_pixels):
            if not still_shifting[i]:
                continue
            new_p = mean_shift_step(p, original_pixels, bandwidth)
            distance = color_dist(np.array(new_p),  np.array(p))
            if distance < epsilon:
                still_shifting[i] = False
            else:
                shifted_pixels[i] = new_p
                converged = False

        iter_count += 1
        if iteration_callback:
            iteration_callback(shifted_pixels, iter_count, in_pixels.shape)

    for i, p in enumerate(shifted_pixels):
        add_point_to_clusters(clusters_centered, clusters_original, p, original_pixels[i], epsilon)

    shifted_pixels = shifted_pixels.reshape(in_pixels.shape)
    return shifted_pixels, clusters_original


def add_point_to_clusters(clusters_centered, clusters_original, clustered_point, original_point, epsilon: float):
    for i, cluster_centered in enumerate(clusters_centered):
        for point in cluster_centered:
            dist = np.linalg.norm(np.array(clustered_point) - np.array(point))
            if dist <= epsilon + 0.05:
                clusters_centered[i].append(clustered_point)
                clusters_original[i].append(original_point)
                return
    # If no suitable cluster was found, create a new cluster
    clusters_centered.append([clustered_point])
    clusters_original.append([original_point])

"""
[x,y] mean_shift(p, original_points):
    shift_x, shift_y, scaleFactor = 0;
    for p_temp in original_points:
        dist = euclidean_dist(p, p_temp)
        weight = kernel(dist, bandwidth)
        shift_x += p_temp[0] * weight
        shift_y += p_temp[1] * weight
        scaleFactor += weight
shift_x = shift_x / scaleFactor
shift_y = shift_y / scaleFactor
return [shift_x, shift_y]
"""
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
    return np.linalg.norm(ref_color - curr_color).astype(float)

def gaussian_weight(dist: float, bandwidth: float) -> float:
    """
    Calculates the weight using a Gaussian kernel based on distance and bandwidth.

    :param dist: Distance between color vectors
    :param bandwidth: Bandwidth parameter for the Gaussian function
    :returns: A weight value based on the Gaussian function
    """
    return np.exp(- (dist ** 2) / (2 * bandwidth ** 2)) # formula from the presentation
    #return (1 / (bandwidth * math.sqrt(2 * math.pi))) * np.exp(-0.5 * (dist / bandwidth) ** 2) # formula from wikipedia

