import cv2 as cv
import matplotlib as mpl
from matplotlib import pyplot as plt, cm

from mean_shift import *

def test_color_monkey_xxs():
    image_filename = "img/color_monkey_small.jpg"
    image = cv.imread(image_filename)
    bandwidth = 0.5
    image_result, clusters = mean_shift_color_pixel(image, bandwidth, epsilon=0.01, max_iter=20, iteration_callback=iteration_callback)
    cv.imshow("Original", image)
    cv.imshow("Mean-Shift Result", image_result)

def test_csv_data():
    data = np.genfromtxt("data/data.csv", delimiter=",")
    result, clusters = mean_shift_color_pixel(data, 1.0, epsilon=0.001, max_iter=2000)
    visualize_meanshift_clusters(result, clusters)

def visualize_meanshift_clusters(center_points, clustered_points):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = mpl.colormaps["viridis"].resampled(len(clustered_points))
    for idx, cluster in enumerate(clustered_points):
        cluster = np.array(cluster)
        ax.scatter(cluster[:, 0], cluster[:, 1], s=50, color=colors(idx), label=f"Cluster {idx + 1}")

    for i, j in center_points:
        ax.scatter(i, j, s=50, c="red", marker="+", label="Center")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()
    fig.savefig("mean_shift_result")


def iteration_callback(shifted_pixels, iter_count, original_shape):
    shifted_pixels = shifted_pixels.reshape(original_shape)
    cv.imshow("shifted_pixels", shifted_pixels)
    cv.waitKey(500)
    print("Iteration count:", iter_count)


test_color_monkey_xxs()
#test_csv_data()

cv.waitKey(0)
cv.destroyAllWindows()
