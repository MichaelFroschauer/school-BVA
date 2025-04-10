import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from visualization import *
#mpl.use('Qt5Agg')

from mean_shift import *

def test_color_monkey_clusters():
    image_filename = "img/color_monkey_xxs.jpg"
    image = cv.imread(image_filename)
    image_result, clusters, cluster_centers = mean_shift_color_pixel(image, 8.0, epsilon=0.001, max_iter=50)
    visualize_meanshift_clusters_based_on_dimensions(clusters, cluster_centers)


def test_color_monkey_animation():

    def callback(shifted_pixels, iter_count, original_shape):
        image_step.append(shifted_pixels.copy())
        print(f"Iteration {iter_count}")

    image_filename = "img/color_monkey_xs.jpg"
    image = cv.imread(image_filename)
    image_step = []
    mean_shift_color_pixel(image, 8.0, epsilon=0.001, max_iter=2000, iteration_callback=callback)
    visualize_as_gif(image_step, "color_monkey_xs.gif", "Mean Shift - Color Monkey")


def test_csv_data_clusters():
    data = np.genfromtxt("data/data.csv", delimiter=",")
    result, clusters, custer_centers = mean_shift_color_pixel(data, 1.0, epsilon=0.001, max_iter=2000)
    visualize_meanshift_clusters_based_on_dimensions(result, clusters)


# tracks = []
# def iteration_callback(shifted_pixels, iter_count, original_shape):
#     tracks.append(shifted_pixels.copy())
#     shifted_pixels = shifted_pixels.reshape(original_shape)
#     #visualize_meanshift_clusters_based_on_dimensions(shifted_pixels)
#     print("Iteration count:", iter_count)


#test_color_monkey_clusters()
test_color_monkey_animation()
#test_csv_data_clusters()


cv.waitKey(0)
cv.destroyAllWindows()
