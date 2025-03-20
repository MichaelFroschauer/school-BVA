# -*- coding: utf-8 -*-
"""
#from https://learnopencv.com/camera-calibration-using-opencv/
"""

import cv2
import numpy as np
import os
import glob
import time


""" Hausübung """
outFilePath = "./recordings/"

# Verzerrungskorrektur auf ein Testbild anwenden
test_img_path = outFilePath + 'img4.png'  # Beispielbild aus der Kalibrierung
test_img = cv2.imread(test_img_path)
h, w = test_img.shape[:2]

# Korrektur der Verzerrung
undistorted_img = cv2.undistort(test_img, mtx, dist, None, mtx)

# Vorher-Nachher-Vergleich anzeigen
cv2.imshow("Original", test_img)
cv2.imshow("Undistorted", undistorted_img)
cv2.imwrite(outFilePath + "img_undistorted.png", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Visualisierung der Verzerrung durch Vektorfeld
map_x, map_y = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (w, h), cv2.CV_32FC1)
flow = np.dstack((map_x - np.arange(w), map_y - np.arange(h)))

# Zeichne Vektorfeld auf Bild
step = 20  # Abstand der Pfeile
for y in range(0, h, step):
    for x in range(0, w, step):
        pt1 = (x, y)
        pt2 = (int(map_x[y, x]), int(map_y[y, x]))
        cv2.arrowedLine(test_img, pt1, pt2, (0, 0, 255), 1, tipLength=0.3)

cv2.imshow("Distortion Vector Field", test_img)
cv2.imwrite(outFilePath + "distortion_vector_field.png", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
