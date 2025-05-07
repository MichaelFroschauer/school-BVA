
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Laden des Bildes
#image = cv2.imread("/home/michael/school/gitclones/2_BVA/0_TestOpenCV/test_image/color_monkey.png", cv2.IMREAD_GRAYSCALE)
#image = cv2.imread("./images/tripod_small.png", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("./images/lena2.png", cv2.IMREAD_GRAYSCALE)

# Berechnung der 2D-Fourier-Transformation
f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)

# Berechnung der spektralen Dichte
magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))
#magnitude_spectrum = np.log(np.abs(f_transform_shifted))
#magnitude_spectrum = np.abs(f_transform_shifted)

# Anzeigen des Originalbildes und der spektralen Dichte
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Magnitude Spectrum')
plt.imshow(magnitude_spectrum, cmap='gray')
plt.show()


x = np.arange(magnitude_spectrum.shape[1])
y = np.arange(magnitude_spectrum.shape[0])
x, y = np.meshgrid(x, y)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, magnitude_spectrum, cmap='viridis')

ax.set_title('2D Surface Plot of Magnitude Spectrum')
ax.set_xlabel('Frequency X')
ax.set_ylabel('Frequency Y')
ax.set_zlabel('Magnitude')

plt.show()

