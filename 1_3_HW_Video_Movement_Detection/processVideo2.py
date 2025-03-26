import cv2
import numpy as np

# Videoquelle (Datei oder Kamera)
inVideoPath = "/home/michael/school/gitclones/2_BVA/1_UE_Lineare_Abbildungssysteme/vtest.avi"
capture = cv2.VideoCapture(inVideoPath)
if not capture.isOpened():
    exit(0)

frameCount = 0
delayInMS = 50  # Zeitverzögerung für Anzeige
train_frames = 50  # Anzahl der Frames für initiale Hintergrundmodellierung
alpha = 0.05  # Gewichtung für gleitenden Durchschnitt

# Background Subtraction mit OpenCV
fgbg_mog2 = cv2.createBackgroundSubtractorMOG2()
fgbg_knn = cv2.createBackgroundSubtractorKNN()

# Initialisierung von Frames
cumulatedFrame = None
heatmap = None

# Kumulierte Bewegungsintensität für die Heatmap
cumulativeMovement = None

# https://github.com/noorkhokhar99/motion-heatmap-opencv/blob/main/heatmap.py

while True:
    ret, frame = capture.read()
    if frame is None:
        break
    frameCopy = frame.copy()

    if frameCount == 0:
        cumulatedFrame = np.zeros_like(frameCopy, dtype=np.float32)
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        cumulativeMovement = np.zeros_like(frameCopy[:, :, 0], dtype=np.float32)

    cumulatedFrame = alpha * frameCopy + (1 - alpha) * cumulatedFrame
    frameCount += 1

    # Durchschnittsbild berechnen
    avgFrame = cumulatedFrame.astype(np.uint8)

    # Background Subtraction mit OpenCV
    fgmask_mog2 = fgbg_mog2.apply(frameCopy)
    fgmask_knn = fgbg_knn.apply(frameCopy)

    # Differenz zwischen aktuellem Frame und Durchschnittsbild
    diffFrame = cv2.absdiff(frameCopy, avgFrame)

    # Motion Maske vom OpenCV Background filter
    _, motionMask = cv2.threshold(fgmask_knn, 30, 255, cv2.THRESH_BINARY)

    # Noise-Filtering mit Morphologie
    #kernel = np.ones((15, 15), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    motionMask = cv2.morphologyEx(motionMask, cv2.MORPH_OPEN, kernel)
    motionMask = cv2.morphologyEx(motionMask, cv2.MORPH_CLOSE, kernel)


    # Hintergrund mit markierten Bewegungsbereichen anzeigen
    movementOverlay = frame.copy()
    movementOverlay[motionMask > 0] = [0, 0, 255]  # Rote Farbe für Bewegung


    # Aktualisiere die kumulierte Bewegungsintensität
    cumulativeMovement += motionMask.astype(np.float32)
    normCumulativeMovement = cv2.normalize(cumulativeMovement, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    # Erstelle eine farbige Heatmap
    heatmapColor = cv2.applyColorMap(normCumulativeMovement, cv2.COLORMAP_HOT)
    mask = (normCumulativeMovement > int(0.1 * 255)).astype(np.uint8)
    heatmapMasked = cv2.bitwise_and(heatmapColor, heatmapColor, mask=mask)
    heatmapOverlay = cv2.addWeighted(frame, 1, heatmapMasked, 0.5, 0)  # Überlagerung mit Heatmap

    # Visualisierung
    cv2.putText(frameCopy, str(frameCount), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.imshow('Original Frame', frameCopy)
    cv2.imshow('Average Frame', avgFrame)
    cv2.imshow('Difference Frame', movementOverlay)
    cv2.imshow('Movement Heatmap', heatmapOverlay)
    cv2.imshow('MOG2 Background Subtraction', fgmask_mog2)
    cv2.imshow('KNN Background Subtraction', fgmask_knn)

    # Warten auf Tastendruck
    keyboard = cv2.waitKey(delayInMS)
    if keyboard == 27:  # ESC zum Beenden
        break

capture.release()
cv2.destroyAllWindows()