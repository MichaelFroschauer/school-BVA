# -*- coding: utf-8 -*-
"""
#from https://learnopencv.com/camera-calibration-using-opencv/
"""

import cv2
import numpy as np
import os
import glob
import time

# Test with checkerboard or circles grid
checkerboard_test_image = True

# Extracting path of individual image stored in a given directory
outFilePath = "/home/michael/school/gitclones/2_BVA/1_2_Camera_Calibration/recordings/"
numOfFrames = 5  # frames to record
numOfValidFrames = 5
timeDelayInSecs = 0.8

# Choose the webcam
inVideoPath = 0 # integrated webcam
#inVideoPath = 2  # external webcam



# Defining the dimensions of checkerboard
if checkerboard_test_image:
    CHECKERBOARD = (6, 9)
else:
    CHECKERBOARD = (4, 11)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None


def showAndSaveImg(img, filePath, fileName):
    cv2.imwrite(filePath + fileName, img)
    cv2.imshow(fileName, img)


capture = cv2.VideoCapture(inVideoPath)
if not capture.isOpened:
    # print('unable to open video: ' + args.input)
    exit(0)

frameCount = 0
validFrameCount = 0

# Blob detector for circles grid
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 200
params.maxArea = 18000
detector = cv2.SimpleBlobDetector_create(params)

while frameCount < numOfFrames and validFrameCount < numOfValidFrames:
    ret, frame = capture.read()
    if frame is None:
        break

    frameCount = frameCount + 1

    frameCopy = frame.copy()
    gray = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2GRAY)

    if checkerboard_test_image:
        # Find the chess board corners
        # If desired number of corners or circles are found in the image then ret = true
        ret, cornersOrCenters = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                          cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    else:
        ret, cornersOrCenters = cv2.findCirclesGrid(gray, CHECKERBOARD, None,
                                                    cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING, blobDetector=detector)

    #    """
    #    If desired number of corner are detected,
    #    we refine the pixel coordinates and display
    #    them on the images of checker board
    #    """
    if ret == True:
        print('checkerboard or circle grid found')

        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, cornersOrCenters, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        frameCopyCB = frameCopy.copy()
        frameCopyCB = cv2.drawChessboardCorners(frameCopyCB, CHECKERBOARD, corners2, ret)

        showAndSaveImg(frameCopy, outFilePath, f'img{str(validFrameCount)}.png')
        showAndSaveImg(frameCopyCB, outFilePath, f'imgCB{str(validFrameCount)}.png')
        validFrameCount = validFrameCount + 1
    else:
        print('checkerboard or circle grid NOT found')
    time.sleep(timeDelayInSecs)


"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret_RMSerr, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# RMS err should be between 0.1 and 1 pixel
if ret_RMSerr < 2.0:
    print('CAMERA CALIBRATED!!! ERR=' + str(ret_RMSerr))
else:
    print('camera calibration FAILED!! ERR=' + str(ret_RMSerr))

print("Camera matrix : \n")
print(mtx)
print("lens distortion : \n")
print(dist)
print('extrinsic positions for ALL detected shapes')
print("ROTATION rvecs : \n")
print(rvecs)
print("TRANSLATION tvecs : \n")
print(tvecs)




# Applying distortion correction to a test image
image_files = sorted(glob.glob(f"{outFilePath}/img[0-9]*.png"))

for idx, test_img_path in enumerate(image_files):

    test_img = cv2.imread(test_img_path)
    h, w = test_img.shape[:2]

    # Correction of the distortion
    testImgCopy = test_img.copy()
    undistorted_img = cv2.undistort(testImgCopy, mtx, dist, None, mtx)

    showAndSaveImg(undistorted_img, outFilePath, f'img{idx}_undistorted.png')

    # Visualization of the distortion by vector field
    map_x, map_y = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (w, h), cv2.CV_32FC1)

    # Draw vector field on image
    step = 20  # Distance of the arrows
    for y in range(0, h, step):
        for x in range(0, w, step):
            pt1 = (x, y)
            pt2 = (int(map_x[y, x]), int(map_y[y, x]))
            cv2.arrowedLine(test_img, pt1, pt2, (0, 0, 255), 1, tipLength=0.3)

    showAndSaveImg(test_img, outFilePath, f'img{idx}_distortion_vector_field.png')

cv2.waitKey(0)
cv2.destroyAllWindows()
