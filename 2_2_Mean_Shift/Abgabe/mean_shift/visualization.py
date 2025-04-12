import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter
from PIL import Image


def visualize_meanshift_clusters_2d(points, center_points=None, image_path=""):
    """
    Visualizes clustered 2D points and optional center points using scatter plots.

    :param points: List of clusters, each cluster is a list of 2D points
    :param center_points: Optional list of 2D center points
    :param image_path: Optional file path to save the resulting plot
    """
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
    fig.savefig(f"{image_path}")


def visualize_meanshift_clusters_3d(points, center_points=None, image_path="", graph_title=""):
    """
    Visualizes clustered 3D points in RGB space with optional center points.

    :param points: List of clusters, each cluster is a list of 3D RGB points
    :param center_points: Optional list of 3D center points
    :param image_path: Optional file path to save the resulting plot
    :param graph_title: Optional title for the plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx, cluster in enumerate(points):
        cluster = [[p[2], p[1], p[0]] for p in cluster]
        cluster = np.array(cluster)
        colors = cluster / 255.0 if cluster.max() > 1.0 else cluster
        ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], s=50, color=colors, label=f"Cluster {idx + 1}")

    if center_points:
        for i, j, k in center_points:
            ax.scatter(i, j, k, s=50, c="red", marker="+", label="Center")

    if len(graph_title) > 0:
        ax.set_title(graph_title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)

    plt.show()
    fig.savefig(f"{image_path}")


def visualize_color_space_as_gif(image_steps, gif_path, gif_title=""):
    """
    Creates a rotating 3D GIF animation of color points in RGB space.

    :param image_steps: List of numpy arrays representing point sets at different steps
    :param gif_path: Output file path for the GIF
    :param gif_title: Optional title shown in the plot
    """
    print(f"Create GIF: {gif_path}")
    writer = PillowWriter(fps=10)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    total_steps = len(image_steps)
    azim = 0
    elev = 30
    azim_step = (azim - 50) / total_steps
    elev_step = (elev - 5) / total_steps

    with writer.saving(fig, gif_path, 300):
        for step_data in image_steps:
            x, y, z = step_data[:, 0], step_data[:, 1], step_data[:, 2]
            colors = np.stack([z, y, x], axis=1) / 255.0
            ax.scatter(x, y, z, s=50, c=colors)

            ax.view_init(azim=azim, elev=elev)
            azim -= azim_step
            #elev -= elev_step

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


def visualize_image_space_as_gif(image_list, output_filename, duration=100):
    """
    Creates a GIF from a sequence of image frames (in image/pixel space).

    :param image_list: List of numpy images (height x width x channels)
    :param output_filename: Path to save the GIF
    :param duration: Frame duration in milliseconds
    """
    pil_images = []

    for img in image_list:
        # Convert BGR to RGB
        if img.shape[2] == 3:
            img = img[..., ::-1]
        pil_image = Image.fromarray(img)
        pil_images.append(pil_image)

    pil_images[0].save(
        output_filename,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0
    )






