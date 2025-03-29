import numpy as np
import random

def apply_kmeans(image, k=4, max_iterations=10):
    """
    Führt K-Means-Clustering auf einem Bild durch.

    :param image: Eingabebild (NumPy-Array)
    :param k: Anzahl der Cluster
    :param max_iterations: Maximale Anzahl von Iterationen
    :return: Segmentiertes Bild mit den Cluster-Farben
    """
    height, width, _ = image.shape

    # Cluster zufällig initialisieren (wähle k zufällige Pixel als Startwerte)
    clusters = np.array(random.choices(image.reshape(-1, 3), k=k), dtype=np.float64)

    for _ in range(max_iterations):
        clusters = update_clusters(image, clusters)

    # Erstelle das segmentierte Bild basierend auf den Cluster-Zuordnungen
    segmented_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            best_cluster_idx = get_best_cluster_idx(image[y, x], clusters)
            segmented_image[y, x] = clusters[best_cluster_idx]

    return segmented_image


def color_dist(ref_color, curr_color):
    """ Berechnet die euklidische Distanz zwischen zwei Farben (RGB). """
    return np.linalg.norm(ref_color - curr_color)


def get_best_cluster_idx(rgb, clusters):
    """ Gibt den Index des nächsten Cluster-Zentrums für eine Farbe zurück. """
    distances = np.linalg.norm(clusters - rgb, axis=1)
    return np.argmin(distances)


def update_clusters(image, clusters):
    """
    Iteriert über alle Pixel und aktualisiert die Cluster-Zentren basierend auf dem Durchschnitt der zugeordneten Farben.

    :param image: NumPy-Array mit Shape (H, W, 3) für RGB-Bild
    :param clusters: NumPy-Array mit Shape (k, 3) für Cluster-Zentren
    :return: Aktualisierte Cluster-Zentren
    """
    height, width, _ = image.shape
    num_clusters = clusters.shape[0]

    # Arrays für die neuen Cluster-Mittelwerte und die Anzahl der zugeordneten Pixel
    new_cluster_sum = np.zeros((num_clusters, 3), dtype=np.float64)
    cluster_counts = np.zeros(num_clusters, dtype=np.int32)

    # Alle Pixel durchgehen und dem nächsten Cluster zuweisen
    for y in range(height):
        for x in range(width):
            pixel = image[y, x]
            best_cluster_idx = get_best_cluster_idx(pixel, clusters)
            new_cluster_sum[best_cluster_idx] += pixel
            cluster_counts[best_cluster_idx] += 1

    # Neue Cluster-Zentren berechnen
    for i in range(num_clusters):
        if cluster_counts[i] > 0:
            clusters[i] = new_cluster_sum[i] / cluster_counts[i]

    return clusters
