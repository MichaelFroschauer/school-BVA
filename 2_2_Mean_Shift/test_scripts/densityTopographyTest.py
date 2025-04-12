import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
mpl.use('Qt5Agg')

# Bild laden und Farben extrahieren
input_file = "/home/michael/school/gitclones/2_BVA/2_2_Mean_Shift/img/color_monkey_xs.jpg"
img = Image.open(input_file).convert("RGB")
pixels = np.array(img).reshape(-1, 3)

# Farbräume normalisieren (0-1)
pixels = pixels / 255.0

# KDE (Kernel Density Estimation) anwenden
kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(pixels)

# Meshgrid für RGB-Raum erzeugen
res = 30
r, g, b = np.linspace(0, 1, res), np.linspace(0, 1, res), np.linspace(0, 1, res)
R, G, B = np.meshgrid(r, g, b)
rgb_samples = np.vstack([R.ravel(), G.ravel(), B.ravel()]).T

# Dichtewerte berechnen
log_densities = kde.score_samples(rgb_samples)
densities = np.exp(log_densities)

# Visualisierung
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(rgb_samples[:, 0], rgb_samples[:, 1], rgb_samples[:, 2],
                c=rgb_samples, s=10, alpha=0.3, linewidths=0,
                marker='o')

# Optional: Dichte als Größe darstellen
# ax.scatter(rgb_samples[:, 0], rgb_samples[:, 1], rgb_samples[:, 2],
#            c=rgb_samples, s=densities*10000, alpha=0.4)

ax.set_xlabel("Red")
ax.set_ylabel("Green")
ax.set_zlabel("Blue")
ax.set_title("3D Color Density Topography (RGB)")
plt.tight_layout()
plt.show()
