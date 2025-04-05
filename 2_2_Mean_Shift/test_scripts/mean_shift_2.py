import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.neighbors import KernelDensity
from mpl_toolkits.mplot3d import Axes3D

def mean_shift_color_pixel(in_pixels, bandwidth):
    # Initialisieren eines zufälligen Shift-Vektors für jedes Pixel
    shifted_pixels = np.copy(in_pixels)

    # Iteriere so lange, bis der Mean-Shift Konvergenz erreicht
    for i in range(len(in_pixels)):
        pixel = in_pixels[i]

        # Berechne den Schwerpunkts-Shift (Mean Shift) für das aktuelle Pixel
        # Hier wird angenommen, dass die benachbarten Pixel im RGB-Raum betrachtet werden
        distances = np.linalg.norm(in_pixels - pixel, axis=1)

        # Wende die Gausssche Gewichtung auf die Distanzen an
        weights = np.exp(- (distances ** 2) / (2 * bandwidth ** 2))

        # Berechne den gewichteten Mittelwert der benachbarten Pixel
        weighted_sum = np.sum(weights[:, None] * in_pixels, axis=0)
        total_weight = np.sum(weights)

        # Berechne den neuen Schwerpunkt des Pixels
        new_pixel = weighted_sum / total_weight

        # Verschiebe das Pixel zum neuen Schwerpunkt
        shifted_pixels[i] = new_pixel

    return shifted_pixels


def mean_shift_with_tracking(pixels, bandwidth, max_iter=10, epsilon=1.0):
    # Jeder Pixel ist ein Punkt im RGB-Raum
    tracks = [ [p.copy()] for p in pixels ]
    for _ in range(max_iter):
        has_converged = True
        new_pixels = []
        for i, p in enumerate(pixels):
            # Finde Punkte im Umkreis
            distances = np.linalg.norm(pixels - p, axis=1)
            weights = np.exp(-(distances ** 2) / (2 * (bandwidth ** 2)))
            # Berechne neuen Schwerpunkt
            weighted_sum = np.sum(pixels * weights[:, np.newaxis], axis=0)
            weight_total = np.sum(weights)
            shifted = weighted_sum / weight_total
            tracks[i].append(shifted.copy())
            if np.linalg.norm(shifted - p) > epsilon:
                has_converged = False
            new_pixels.append(shifted)
        pixels = np.array(new_pixels)
        if has_converged:
            break
    return np.array(new_pixels), tracks


def animate_rgb_and_image(tracks, image_shape):
    fig = plt.figure(figsize=(12, 6))
    ax_img = fig.add_subplot(1, 2, 1)
    ax_rgb = fig.add_subplot(1, 2, 2, projection='3d')

    num_frames = len(tracks[0])

    def update(frame):
        ax_img.clear()
        ax_rgb.clear()

        current_pixels = np.array([track[frame] for track in tracks])
        image = current_pixels.reshape(image_shape).astype(np.uint8)
        ax_img.imshow(image)
        ax_img.set_title(f"Iteration {frame}")

        r, g, b = current_pixels[:, 0], current_pixels[:, 1], current_pixels[:, 2]
        ax_rgb.scatter(r, g, b, c=current_pixels / 255.0, s=2)
        ax_rgb.set_xlim(0, 255)
        ax_rgb.set_ylim(0, 255)
        ax_rgb.set_zlim(0, 255)
        ax_rgb.set_xlabel("R")
        ax_rgb.set_ylabel("G")
        ax_rgb.set_zlabel("B")
        ax_rgb.set_title("RGB Space")

    ani = FuncAnimation(fig, update, frames=num_frames, interval=500)
    plt.show()


def colour_dist(ref_color, curr_color):
    return np.linalg.norm(np.array(ref_color) - np.array(curr_color))

def gaussian_weight(dist, bandwidth):
    return np.exp(- (dist ** 2) / (2 * bandwidth ** 2))

def visualize_3d_color_density(pixels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Trenne RGB in 3 Listen
    r = pixels[:, 0]
    g = pixels[:, 1]
    b = pixels[:, 2]

    # Erstelle 3D-Scatter-Plot
    ax.scatter(r, g, b, c=np.array(pixels) / 255, s=1)

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    plt.show()


def visualize_2d_image(pixels, image_shape):
    # Erstelle ein Bild aus den verschobenen Pixeln
    image = pixels.reshape(image_shape)

    # Zeige das Bild an
    cv2.imshow('Mean Shift Clustering', image)
    cv2.waitKey(1)  # Eine kleine Pause, um die Animation zu sehen


def mean_shift_clustering(image, bandwidth):
    # Umwandlung des Bildes in ein Array von RGB-Pixeln
    pixels = image.reshape((-1, 3))  # Bild in ein flaches Array umwandeln

    # Wende den Mean-Shift-Algorithmus an
    shifted_pixels = mean_shift_color_pixel(pixels, bandwidth)

    # Visualisiere die 3D-Farbdichte
    visualize_3d_color_density(shifted_pixels)

    # Visualisiere das 2D-Bild während der Iterationen
    visualize_2d_image(shifted_pixels, image.shape)

    # Rückgabe des Bildes mit den verschobenen Pixeln
    return shifted_pixels.reshape(image.shape)


def count_clusters(pixels, threshold=5):
    cluster_centers = []
    for pixel in pixels:
        found_cluster = False
        for center in cluster_centers:
            if colour_dist(center, pixel) < threshold:
                found_cluster = True
                break
        if not found_cluster:
            cluster_centers.append(pixel)

    return len(cluster_centers)

def plot_rgb_density(image_rgb, bandwidth=20.0, sample=10000):
    # Farben als Punkte im RGB-Raum
    pixels = image_rgb.reshape(-1, 3)
    if len(pixels) > sample:
        idx = np.random.choice(len(pixels), sample, replace=False)
        pixels = pixels[idx]

    # KDE trainieren
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(pixels)

    # Grid zum Auswerten der Dichte
    grid_size = 30
    r = np.linspace(0, 255, grid_size)
    g = np.linspace(0, 255, grid_size)
    b = np.linspace(0, 255, grid_size)
    rr, gg, bb = np.meshgrid(r, g, b)
    grid_samples = np.vstack([rr.ravel(), gg.ravel(), bb.ravel()]).T

    # Dichte berechnen
    log_density = kde.score_samples(grid_samples)
    density = np.exp(log_density)

    # Schwelle für hohe Dichte
    threshold = np.percentile(density, 97)

    # Punkte mit hoher Dichte plotten
    high_density_points = grid_samples[density > threshold]
    high_density_vals = density[density > threshold]

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = high_density_points / 255.0
    ax.scatter(
        high_density_points[:, 0],
        high_density_points[:, 1],
        high_density_points[:, 2],
        c=colors,
        s=10 * (high_density_vals / max(high_density_vals)),
        alpha=0.6
    )
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.set_title('3D RGB Color Density Topography')
    plt.show()


image = cv2.imread("../img/color_monkey_small.jpg")

img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plot_rgb_density(img, bandwidth=20.0)

# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# pixels = image_rgb.reshape(-1, 3)
# clustered, tracks = mean_shift_with_tracking(pixels, bandwidth=20)
# animate_rgb_and_image(tracks, image_rgb.shape)

# #bandwidths = [5, 10, 20]
# bandwidths = [100]
# for bandwidth in bandwidths:
#     image_copy = image.copy()
#     clustered_image = mean_shift_clustering(image_copy, bandwidth)
#     num_clusters = count_clusters(clustered_image)
#     print(f"Anzahl der Cluster bei Bandbreite {bandwidth}: {num_clusters}")
#
#     clustered_image_bgr = cv2.cvtColor(clustered_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
#
#     # Anzeige
#     cv2.imshow(f"Mean Shift Result {bandwidth}", clustered_image_bgr)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     # Oder speichern
#     cv2.imwrite(f"mean_shift_result_{bandwidth}.jpg", clustered_image_bgr)
#
#     pixels = clustered_image.reshape(-1, 3)
#     visualize_3d_color_density(pixels)


