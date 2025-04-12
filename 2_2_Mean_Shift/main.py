import os
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from visualization import *
#mpl.use('Qt5Agg')

from mean_shift import *

def test_csv_data_clusters():
    data = np.genfromtxt("data/data.csv", delimiter=",")
    result, clusters, custer_centers = mean_shift_color_pixel(data, 1.0, epsilon=0.001, max_iter=2000)
    visualize_meanshift_clusters_2d(clusters, custer_centers, f"{output_path}/mean_shift_result_2d.png")

def test_color_monkey_clusters():
    image_filename = "img/color_monkey_xxs.jpg"
    image = cv.imread(image_filename)
    image_result, clusters, cluster_centers = mean_shift_color_pixel(image, 8.0, epsilon=0.001, max_iter=50)
    visualize_meanshift_clusters_3d(clusters, cluster_centers, f"{output_path}/mean_shift_result_3d.png")


def test_mean_shift_with_animation(image_filename, bandwidth, epsilon=1.0, max_iter=200, output_path="output", title=""):
    def callback(step_data):
        image_step_cs.append(step_data.shifted_pixels.copy())
        image_step_is.append(step_data.shifted_pixels.copy().reshape(step_data.original_image_shape))
        step_data.print_step_results()
        # Optionally visualize the image here directly

    image = cv.imread(f"./img/{image_filename}")
    image_step_cs = []
    image_step_is = []
    name, _ = os.path.splitext(image_filename)
    print(f"Start mean shift clustering with image: {image_filename} and bandwidth: {bandwidth} for {image.shape[0] * image.shape[1]} pixels")
    image_result, clusters, cluster_centers  = mean_shift_color_pixel(image, bandwidth, epsilon=epsilon, max_iter=max_iter, iteration_callback=callback)
    visualize_color_space_as_gif(image_step_cs, f"{output_path}/{name}_{bandwidth}_cs.gif",f"{title}")
    visualize_image_space_as_gif(image_step_is, f"{output_path}/{name}_{bandwidth}_im.gif")

    cv.imwrite(f"{output_path}/{name}_{bandwidth}_im.png", image_result)
    visualize_meanshift_clusters_3d(list(image), None, f"{output_path}/{name}_{bandwidth}_cs_before.png",
                                    graph_title=f"{name} - Before Mean Shift - BW {bandwidth}")
    visualize_meanshift_clusters_3d(list(image_result), None, f"{output_path}/{name}_{bandwidth}_cs_after.png",
                                    graph_title=f"{name} - After Mean Shift - BW {bandwidth}")


output_path = "/home/michael/school/gitclones/2_BVA/2_2_Mean_Shift/output"

test_color_monkey_clusters()
test_csv_data_clusters()

test_mean_shift_with_animation("color_monkey_xxs.jpg", 15.0, 0.001, 50, title="Mean Shift Color Space XXS - Color Monkey")
test_mean_shift_with_animation("color_monkey_xs.jpg", 30.0, 0.1, 40, title="Color Space XS - Color Monkey - BW 30")
test_mean_shift_with_animation("color_monkey_xs.jpg", 20.0, 0.1, 40, title="Color Space XS - Color Monkey - BW 20")
test_mean_shift_with_animation("color_monkey_xs.jpg", 10.0, 0.1, 40, title="Color Space XS - Color Monkey - BW 10")
test_mean_shift_with_animation("color_monkey_xs.jpg",  5.0, 0.1, 40, title="Color Space XS - Color Monkey - BW 5")

test_mean_shift_with_animation("bird_xs.png", 40.0, 0.1, 2000, title="Color Space XS - Bird - BW 40")
test_mean_shift_with_animation("bird_xs.png", 10.0, 0.1, 80, title="Color Space XS - Bird - BW 10")
test_mean_shift_with_animation("bird_xs.png",  5.0, 0.1, 2000, title="Color Space XS - Bird - BW 5")

test_mean_shift_with_animation("color_monkey_s.jpg", 20.0, 1.0, 200, title="Mean Shift Color Space S - Color Monkey")

#test_mean_shift_with_animation("boats_s.png", 20.0, 1.0, 200, title="Mean Shift Color Space - Boats")
#test_mean_shift_with_animation("bird_s.png", 20.0, 1.0, 200, title="Mean Shift Color Space - Bird")


cv.waitKey(0)
cv.destroyAllWindows()
