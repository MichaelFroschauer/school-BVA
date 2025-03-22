# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:05:18 2020

@author: P21702
"""
#adapted from https://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html

import cv2 
import numpy as np

#inVideoPath = "/home/michael/school/gitclones/2_BVA/3_HW_Video_Movement_Detection/vtest.avi"
inVideoPath = "/home/michael/school/gitclones/2_BVA/3_HW_Video_Movement_Detection/test_video_cards.mp4"
#inVideoPath = 0 # 0 = live camera

capture = cv2.VideoCapture(inVideoPath)
if not capture.isOpened:
    print('unable to open: ' + inVideoPath)
    exit(0)

frameCount = 0
delayInMS = 100

# Background Subtraction mit OpenCV
fgbgMog2 = cv2.createBackgroundSubtractorMOG2()
fgbgKnn = cv2.createBackgroundSubtractorKNN()


def applyOpenCvBackgroundSubtraction(cframe, bgSubtractor):
    # test background subtraction with OpenCV functions
    fgMask = bgSubtractor.apply(cframe)
    _, motionMask = cv2.threshold(fgMask, 50, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # filter kernel
    motionMask = cv2.morphologyEx(motionMask, cv2.MORPH_OPEN, kernel)  # remove single pixel artefacts
    motionMask = cv2.morphologyEx(motionMask, cv2.MORPH_CLOSE, kernel)  # close created gaps
    movementOverlay = cframe.copy()
    movementOverlay[motionMask > 0] = [0, 0, 255]
    return (movementOverlay, motionMask)

def createHeatmap(cumulativeMovement):
    normCumulativeMovement = cv2.normalize(cumulativeMovement, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    normCumulativeMovement = cv2.morphologyEx(normCumulativeMovement, cv2.MORPH_DILATE, kernel)  # remove clearly visible outlines
    heatmapColor = cv2.applyColorMap(normCumulativeMovement, cv2.COLORMAP_HOT)
    heatmapOverlay = cv2.addWeighted(frameCopy, 1, heatmapColor, 0.7, 0)
    return heatmapOverlay

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    frameCopy = frame.copy() # deep copy, atomar

    # calc cumulated frame (allocate in case of frame #0)
    if frameCount == 0:
        # initializing the background
        cumulatedFrame = np.zeros(frameCopy.shape)
        cumulatedFrame += frameCopy
        frameCount = 1
        cumulativeMovement = np.zeros(frameCopy.shape[:2], dtype=np.float32)
        cumulativeMovementMog2 = np.zeros(frameCopy.shape[:2], dtype=np.float32)
        cumulativeMovementKnn = np.zeros(frameCopy.shape[:2], dtype=np.float32)
    else:
        cumulatedFrame += frameCopy
        frameCount = frameCount + 1

    """" Background Detection """
    # calculate average frame
    avgFrame = cumulatedFrame / (frameCount, frameCount, frameCount)
    avgFrame = avgFrame.astype(np.uint8)

    # get the difference between the current frame and the average frame
    diffFrame = cv2.absdiff(frameCopy, avgFrame)

    """" Movement Detection """
    # create a motion mask by defining a threshold for the difference frame
    _, motionMask = cv2.threshold(diffFrame, 50, 255, cv2.THRESH_BINARY)

    # create and apply filter to reduce noise
    #kernel = np.ones((3, 3), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))   # filter kernel
    motionMask = cv2.morphologyEx(motionMask, cv2.MORPH_OPEN, kernel)     # remove single pixel artefacts
    motionMask = cv2.morphologyEx(motionMask, cv2.MORPH_CLOSE, kernel)    # close created gaps

    # add the motion mask as red to the original frame
    movementOverlay = frame.copy()
    movementOverlay[motionMask[..., 0] > 0] = [0, 0, 255]

    """" Heatmap """
    # add up the motion mask for the heatmap
    cumulativeMovement += motionMask[..., 0].astype(np.float32)
    normCumulativeMovement = cv2.normalize(cumulativeMovement, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # apply filter to the heatmap overlay
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # filter kernel
    normCumulativeMovement = cv2.morphologyEx(normCumulativeMovement, cv2.MORPH_DILATE, kernel)  # remove clearly visible outlines
    #normCumulativeMovement = cv2.morphologyEx(normCumulativeMovement, cv2.MORPH_OPEN, kernel)   # remove single pixel artefacts
    #normCumulativeMovement = cv2.morphologyEx(normCumulativeMovement, cv2.MORPH_CLOSE, kernel)  # close created gaps

    heatmapColor = cv2.applyColorMap(normCumulativeMovement, cv2.COLORMAP_HOT)
    heatmapOverlay = cv2.addWeighted(frameCopy, 1, heatmapColor, 0.7, 0)

    """" OpenCV Background Detection """
    # test background subtraction with OpenCV functions
    movementMog2, motionMaskMog2 = applyOpenCvBackgroundSubtraction(frameCopy, fgbgMog2)
    movementKnn, motionMaskKnn = applyOpenCvBackgroundSubtraction(frameCopy, fgbgKnn)
    cumulativeMovementMog2 += motionMaskMog2.astype(np.float32)
    cumulativeMovementKnn += motionMaskKnn.astype(np.float32)
    heatmapOverlayMog2 = createHeatmap(cumulativeMovementMog2)
    heatmapOverlayKnn = createHeatmap(cumulativeMovementKnn)


    # show all frames in a separate window
    cv2.imshow('Original Frame', frameCopy)
    cv2.imshow('Average Frame', avgFrame)
    cv2.imshow('Difference Frame', diffFrame)
    cv2.imshow('Movement Frame', movementOverlay)
    cv2.imshow('Movement Heatmap', heatmapOverlay)
    cv2.imshow('Movement Frame with OpenCV MOG2', movementMog2)
    cv2.imshow('Movement Frame with OpenCV KNN', movementKnn)
    cv2.imshow('Movement Heatmap with OpenCV MOG2', heatmapOverlayMog2)
    cv2.imshow('Movement Heatmap with OpenCV Knn', heatmapOverlayKnn)

    # print statistics
    maxVal = np.max(cumulatedFrame)
    avgVal = np.average(cumulatedFrame)
    print("iter " + str(frameCount) + " max= " + str(maxVal) + " avg= " + str(avgVal))

    # average value statistics
    maxVal = np.max(avgFrame)
    avgVal = np.average(avgFrame)
    print("AVG iter " + str(frameCount) + " max= " + str(maxVal) + " avg= " + str(avgVal))

    # print current frame count
    cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv2.putText(frame, str(frameCount), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    keyboard = cv2.waitKey(delayInMS)
