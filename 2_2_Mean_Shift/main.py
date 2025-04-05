import cv2 as cv
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from mean_shift import *

def test_color_monkey_xxs():
    image_filename = "img/color_monkey_xxs.jpg"
    image = cv.imread(image_filename)
    bandwidth = 8.0
    image_result, clusters, cluster_centers = mean_shift_color_pixel(image, bandwidth, epsilon=0.001, max_iter=50, iteration_callback=iteration_callback)
    cv.imshow("Original", image)
    cv.imshow("Mean-Shift Result", image_result)
    visualize_based_on_dimensions(clusters, cluster_centers)

def test_csv_data():
    data = np.genfromtxt("data/data.csv", delimiter=",")
    result, clusters, custer_centers = mean_shift_color_pixel(data, 1.0, epsilon=0.001, max_iter=2000)
    visualize_based_on_dimensions(result, clusters)

def visualize_based_on_dimensions(points, center_points=None):
    if len(points[0][0]) == 2:
        print("Using 2D visualization")
        visualize_meanshift_clusters_2d(points, center_points)
    elif len(points[0][0]) == 3:
        print("Using 3D visualization")
        visualize_meanshift_clusters_3d(points, center_points)
    else:
        print("Unsupported dimensions")

def visualize_meanshift_clusters_2d(points, center_points=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = mpl.colormaps["viridis"].resampled(len(points))
    for idx, cluster in enumerate(points):
        cluster = np.array(cluster)
        ax.scatter(cluster[:, 0], cluster[:, 1], s=50, color=colors(idx), label=f"Cluster {idx + 1}")

    if center_points:
        for i, j in center_points:
            ax.scatter(i, j, s=50, c="red", marker="+", label="Center")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()
    fig.savefig("mean_shift_result_2d.png")


def visualize_meanshift_clusters_3d(points, center_points=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = mpl.colormaps["viridis"].resampled(len(points))
    for idx, cluster in enumerate(points):
        cluster = np.array(cluster)
        ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], s=50, color=colors(idx), label=f"Cluster {idx + 1}")

    if center_points:
        for i, j, k in center_points:
            ax.scatter(i, j, k, s=50, c="red", marker="+", label="Center")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    fig.savefig("mean_shift_result_3d.png")


tracks = []
def iteration_callback(shifted_pixels, iter_count, original_shape):
    shifted_pixels = shifted_pixels.reshape(original_shape)
    #cv.imshow("shifted_pixels", shifted_pixels)
    #cv.waitKey(10)
    #visualize_based_on_dimensions(shifted_pixels)
    tracks.append(shifted_pixels)
    print("Iteration count:", iter_count)

test_color_monkey_xxs()
#test_csv_data()

cv.waitKey(0)
cv.destroyAllWindows()
