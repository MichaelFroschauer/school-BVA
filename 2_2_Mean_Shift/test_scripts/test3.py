import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.datasets import load_sample_image
from scipy.ndimage import gaussian_filter
import cv2 as cv

# Beispielbild laden (China Lake Bild von sklearn)
#china = load_sample_image("china.jpg")

#image_filename = "bird_xs.png"
image_filename = "color_monkey_xs.jpg"
img = cv.imread(f"/home/michael/school/gitclones/2_BVA/2_2_Mean_Shift/img/{image_filename}")
#img = img[::10, ::10]  # Bild verkleinern für bessere Performance

# RGB-Werte extrahieren
pixels = img.reshape(-1, 3)

# Histogramm der RGB-Verteilung (hier nur R und G, B fix auf ~128 für 2D-Darstellung)
# -> Eine Art Slice durch den RGB-Würfel
r = pixels[:, 0]
g = pixels[:, 1]
b = pixels[:, 2]

# B in Bins klassifizieren, z. B. [120, 136]
b_mask = (b > 120) & (b < 136)
r_filtered = r[b_mask]
g_filtered = g[b_mask]

# Histogramm (2D Dichte) berechnen
bins = 32
hist, xedges, yedges = np.histogram2d(r_filtered, g_filtered, bins=bins, range=[[0, 256], [0, 256]])
hist = gaussian_filter(hist, sigma=1.2)  # Glätten für schönere Oberfläche

# Meshgrid erstellen
x_centers = (xedges[:-1] + xedges[1:]) / 2
y_centers = (yedges[:-1] + yedges[1:]) / 2
X, Y = np.meshgrid(x_centers, y_centers)

# Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, hist.T, cmap=cm.viridis, edgecolor='k', linewidth=0.1)

ax.set_title("3D Color Density Topography (R-G Plane, B ≈ 128)")
ax.set_xlabel("Red")
ax.set_ylabel("Green")
ax.set_zlabel("Density")

plt.tight_layout()
plt.show()
