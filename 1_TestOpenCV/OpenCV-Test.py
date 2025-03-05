 # -*- coding: utf-8 -*-
"""
Test OpenCV

read Image and basic image processing
"""

import cv2 #OpenCv
import numpy as np #Numpy
import matplotlib.pyplot as plt #charting


# RGB
inPath = "./test_image/affe.png"
outPathGray = "./test_image/gray_monkey.png"
#load with OpenCV
img = cv2.imread(inPath)
print("shape of image " + str(img.shape))

# Gray Value
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("shape of gray image " + str(grayImg.shape))

# Image statistics
minVal = np.min(grayImg)
maxVal = np.max(grayImg)
avgVal = np.mean(grayImg)
print("min = " + str(minVal) + " maxVal = " + str(maxVal) + " avg= " + str(avgVal))

# show image with open cv
# cv2.imshow("gray image", grayImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# show image with mathplot
# imgplot = plt.imshow(grayImg)
# plt.show()
# plt.imshow(grayImg, cmap="hot")
# plt.show() # show images with OpenCV

# write image to file system
cv2.imwrite(outPathGray, grayImg)
