import cv2
import numpy as np
from ultralytics import YOLO

# YOLO Modell laden
model = YOLO("yolo11n.pt")

# Bild laden
image_path = "./img/ball_1.jpg"
image = cv2.imread(image_path)

# Objekterkennung durchführen
results = model(image_path)

for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()  # Bounding-Box-Koordinaten

    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)

        # Bounding Box ausschneiden
        roi = image[y_min:y_max, x_min:x_max]

        # In HSV-Farbraum konvertieren
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Farbfilter für Ballsegmentierung (z. B. Orange für Basketball, Weiß für Fußball)
        lower_bound = np.array([10, 100, 100])  # Anpassen für deine Farbe
        upper_bound = np.array([25, 255, 255])  # Anpassen für deine Farbe

        # Maske erstellen
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Konturen erkennen
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Größte Kontur extrahieren (Ball)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(roi, [largest_contour], -1, (0, 255, 0), 2)  # Kontur zeichnen

        # Originalbild aktualisieren
        image[y_min:y_max, x_min:x_max] = roi

# Bild mit Segmentierung anzeigen
cv2.imshow("Segmented Ball", image)
cv2.imwrite("./output/segmented_ball.jpg", image)


offsets = []

for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()

    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        roi = image[y_min:y_max, x_min:x_max]

        # Bounding Box Zentrum berechnen
        bbox_center_x = (x_min + x_max) // 2
        bbox_center_y = (y_min + y_max) // 2

        # In HSV umwandeln
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([10, 100, 100])  # Anpassen für deine Farbe
        upper_bound = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Konturen suchen
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Mittelpunkt der Kontur berechnen
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                ball_center_x = int(M["m10"] / M["m00"]) + x_min
                ball_center_y = int(M["m01"] / M["m00"]) + y_min

                # Differenz speichern
                offsets.append((ball_center_x - bbox_center_x, ball_center_y - bbox_center_y))

                # Zentren zeichnen
                cv2.circle(image, (bbox_center_x, bbox_center_y), 5, (255, 0, 0), -1)  # Blau = Bounding Box Zentrum
                cv2.circle(image, (ball_center_x, ball_center_y), 5, (0, 0, 255), -1)  # Rot = Ballzentrum

# Statistik berechnen
offsets = np.array(offsets)
mean_offset_x = np.mean(offsets[:, 0])
mean_offset_y = np.mean(offsets[:, 1])
std_dev_x = np.std(offsets[:, 0])
std_dev_y = np.std(offsets[:, 1])

print(f"Mean Offset X: {mean_offset_x:.2f}, Mean Offset Y: {mean_offset_y:.2f}")
print(f"Standard Deviation X: {std_dev_x:.2f}, Standard Deviation Y: {std_dev_y:.2f}")

# Bild anzeigen
cv2.imshow("Comparison of Centers", image)
cv2.imwrite("./output/comparison_of_centers.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()