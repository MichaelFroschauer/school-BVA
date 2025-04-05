import cv2
from kMeansClusting import *


image = cv2.imread("landscape_small.jpg")
image_copy = image.copy()
k_image = apply_kmeans(image_copy, 20, 5)
cv2.imshow("k_image_20_5", k_image)


cv2.waitKey(0)
cv2.destroyAllWindows()