# Kamera-Kalibrierung mit OpenCV

## Aufgabe 4: Implementierung der Kamera-Kalibrierung

### Zielsetzung
Die Kamera-Kalibrierung soll mit OpenCV durchgeführt werden. Dabei wird ein Schachbrettmuster oder ein Kreisraster zur Kalibrierung verwendet. Ziel ist es, die intrinsischen, extrinsischen und Verzerrungsparameter der Kamera zu bestimmen und diese für eine spätere Bildkorrektur zu nutzen.

### Vorgehensweise
1. **Erfassen von Kalibrierungsbildern:**
   - Bilder eines Schachbrettmusters wurden mit einer externen Webcam aufgenommen.
   - Es wurden 10 Bilder erfasst, wobei mindestens eines erfolgreich erkannt werden musste.

2. **Erkennung des Musters:**
   - Die Funktion `cv2.findChessboardCorners` wurde verwendet, um die Ecken des Schachbretts zu identifizieren.
   - Alternativ wurde `cv2.findCirclesGrid` für die Erkennung eines Kreisrasters getestet, jedoch ohne Erfolg.

3. **Berechnung der Kameraparameter:**
   - Die bekannten 3D-Weltkoordinaten wurden mit den erkannten 2D-Pixelkoordinaten in den Bildern verknüpft.
   - Mit der Funktion `cv2.calibrateCamera` wurden die folgenden Parameter berechnet:
     - **Kameramatrix (intrinsische Parameter)**
     - **Verzerrungskoeffizienten**
     - **Rotations- und Translationsvektoren (extrinsische Parameter)**

### Ergebnisse der Kalibrierung

Die Kamera-Kalibrierung war erfolgreich. Die berechneten Parameter sind:

**Root Mean Square Error (RMS):** 0.1005

**Kameramatrix:**
```
[[553.01889865   0.         332.11789761]
 [  0.         555.80842144 279.65187391]
 [  0.           0.           1.        ]]
```

**Verzerrungskoeffizienten:**
```
[[ 0.00755846 -0.34561566  0.00227644 -0.0043784   1.98345472]]
```

**Extrinsische Parameter:**
```
ROTATION:
[[-0.03872679]
 [-0.0728116 ]
 [-1.53204618]]
TRANSLATION:
[[-4.27172153]
 [ 3.58962755]
 [13.62758044]]
```

---

## Aufgabe 6: Verzerrungskorrektur und Visualisierung

### Zielsetzung
Mit den ermittelten Kalibrierungsergebnissen soll eine Verzerrungskorrektur auf Videoframes angewendet werden. Zusätzlich soll eine Visualisierung der Verzerrung vor und nach der Korrektur erfolgen.

### Vorgehensweise
1. **Anwenden der Verzerrungskorrektur:**
   - Mit der Funktion `cv2.undistort` wurde die Korrektur auf Testbilder angewendet.
   - Die korrigierten Bilder wurden gespeichert und angezeigt.

2. **Visualisierung der Verzerrung:**
   - Ein Vektorfeld wurde zur Darstellung der Verzerrung vor der Korrektur genutzt.
   - Die Funktion `cv2.initUndistortRectifyMap` wurde verwendet, um eine Mapping-Tabelle für die Pixelverschiebung zu erstellen.
   - Die Verschiebung der Pixel wurde mit roten Pfeilen auf dem Originalbild dargestellt.

### Ergebnisse
Die Verzerrungskorrektur konnte erfolgreich durchgeführt werden. Die Visualisierung zeigt deutlich, wie stark die Verzerrung vor der Korrektur war und welche Bereiche besonders betroffen sind.

**Visualisierungsansätze:**
- **Vektorfeld:** Darstellung der Pixelverschiebung mit Pfeilen
- **Checkerboard-Farbdifferenz:** (Optional für eine alternative Darstellung)

### Fazit
Die durchgeführte Kamera-Kalibrierung liefert präzise Verzerrungskorrekturen. Die erstellte Visualisierung bietet eine anschauliche Möglichkeit, die Verzerrung und deren Korrektur zu verstehen. Durch den Vergleich mit unterschiedlichen Kameras können weitere Erkenntnisse über die Qualität der Kalibrierung gewonnen werden.

