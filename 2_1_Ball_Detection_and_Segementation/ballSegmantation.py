import os
import pickle

import cv2
from ultralytics import YOLO
import numpy as np


def scale_image(image_to_scale, scale_percent):
    width = int(image_to_scale.shape[1] * scale_percent)
    height = int(image_to_scale.shape[0] * scale_percent)
    new_size = (width, height)
    resized_image = cv2.resize(image_to_scale, new_size, interpolation=cv2.INTER_AREA)
    return resized_image


# YOLO Modell laden
model = YOLO("yolo11n.pt")

# Bild laden
image_path = "./img/ball_5.jpg"
image = cv2.imread(image_path)

# Objekterkennung durchführen
results = None
result_file = f"./output/{image_path.split('/')[-1].split('.')[0]}.pkl"
if os.path.exists(result_file):
    with open(result_file, "rb") as f:
        results = pickle.load(f)
    print("Geladene Ergebnisse aus Datei.")
else:
    results = model(image_path)
    with open(result_file, "wb") as f:
        pickle.dump(results, f)
    print("Neue Ergebnisse gespeichert.")


# Klassenindex zu Klassenname (falls Modell Standard-COCO Klassen nutzt)
class_names = model.names  # Gibt eine Liste der Klassennamen zurück

ball_boxes = []
image_bounding_box = image.copy()

# Ergebnisse verarbeiten und Bounding Boxes zeichnen
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()  # Bounding-Box-Koordinaten
    confidences = r.boxes.conf.cpu().numpy()  # Konfidenzwerte
    class_ids = r.boxes.cls.cpu().numpy().astype(int)  # Klassen-IDs

    for box, conf, class_id in zip(boxes, confidences, class_ids):
        x_min, y_min, x_max, y_max = map(int, box)
        label = f"{class_names[class_id]}: {conf:.2f}"  # Name + Konfidenz

        # Bounding Box zeichnen
        cv2.rectangle(image_bounding_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Text mit Objektnamen + Konfidenz einfügen
        cv2.putText(image_bounding_box, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

        if class_names[class_id] == "sports ball":
            ball_boxes.append([x_min, y_min, x_max, y_max])


image_segmented = image.copy()
# Farbmaske für den Ball (z. B. Weiß, Schwarz, Gelb bei einem Fußball)
lower_white = np.array([0, 100, 100])
upper_white = np.array([255, 255, 255])

for x_min, y_min, x_max, y_max in ball_boxes:

    roi = image_segmented[y_min:y_max, x_min:x_max]
    mask = cv2.inRange(roi, lower_white, upper_white)
    mask_ball = cv2.bitwise_not(mask)

    # **DISRUPTION-ERUPTION FILTER (Morphologische Operationen)**
    #kernel = np.ones((5, 5), np.uint8)  # 5x5 Kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_opened = cv2.morphologyEx(mask_ball, cv2.MORPH_OPEN, kernel)  # Entfernt Rauschen
    mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)  # Schließt Lücken

    # Maske auf das Bild anwenden
    #result_roi = cv2.bitwise_and(roi, roi, mask=mask_cleaned)
    #ballOverlay = image.copy()
    #a = mask_cleaned[..., 0]
    #ballOverlay[a > 0] = [0, 0, 255]
    #image_segmented[y_min:y_max, x_min:x_max] = ballOverlay

    overlay_color = (0, 0, 255)  # BGR-Wert für Rot
    alpha = 1.0  # Transparenz (0.0 = voll transparent, 1.0 = voll deckend)
    overlay = np.zeros_like(roi, dtype=np.uint8)
    overlay[:] = overlay_color
    ball_area = cv2.bitwise_and(overlay, overlay, mask=mask_cleaned)
    #result_roi[:] = overlay_color
    #roi_with_overlay = cv2.addWeighted(result_roi, alpha, roi, 1 - alpha, 0)
    roi_with_overlay = cv2.addWeighted(roi, 1 - alpha, ball_area, alpha, 0)

    image_segmented[y_min:y_max, x_min:x_max] = roi_with_overlay


resized_result = scale_image(image_segmented, 0.6)
cv2.imshow("Segmentierter Ball mit Disruption-Eruption", resized_result)

# Bild anzeigen
resized_image = scale_image(image_bounding_box, 0.8)
cv2.imshow("Detected Objects", resized_image)
cv2.imwrite("./output/detected_ball.jpg", image_bounding_box)
cv2.waitKey(0)
cv2.destroyAllWindows()
