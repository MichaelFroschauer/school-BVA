import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter


def visualize_meanshift_clusters_based_on_dimensions(points, center_points=None):
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


def visualize_as_gif(image_steps, gif_name, gif_title=""):
    print(f"Create GIF: {gif_name}")
    writer = PillowWriter(fps=10)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # azim 0 -> -50
    # elev 30 -> 5
    total_steps = len(image_steps)
    azim = 0
    elev = 30
    azim_step = (azim - 50) / total_steps
    elev_step = (elev - 5) / total_steps

    with writer.saving(fig, gif_name, 300):
        for step_data in image_steps:
            x, y, z = step_data[:, 0], step_data[:, 1], step_data[:, 2]
            colors = np.stack([x, y, z], axis=1) / 255.0
            ax.scatter(x, y, z, s=50, c=colors)

            ax.view_init(azim=azim, elev=elev)
            azim -= azim_step
            elev -= elev_step

            if len(gif_title) > 0:
                ax.set_title(gif_title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 255)
            ax.set_zlim(0, 255)

            writer.grab_frame()
            plt.cla()