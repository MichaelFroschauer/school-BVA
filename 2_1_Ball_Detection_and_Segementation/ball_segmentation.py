import os
import pickle
import cv2
from ultralytics import YOLO
from image_processing import *


# Load YOLO model
model = YOLO("yolo11n.pt")

# Load image
image_path = "./img/ball_7.jpg"
image = cv2.imread(image_path)

# Perform object detection
yolo_results = None
result_file = f"./output/{image_path.split('/')[-1].split('.')[0]}.pkl"
if os.path.exists(result_file):
    with open(result_file, "rb") as f:
        yolo_results = pickle.load(f)
    print(f"Use saved yolo results from file: {result_file}")
else:
    yolo_results = model(image_path)
    with open(result_file, "wb") as f:
        pickle.dump(yolo_results, f)
    print(f"Save yolo results to file: {result_file}")


class_names = model.names
image_bounding_box = image.copy()
image_bounding_box, ball_boxes = process_detections(image_bounding_box, yolo_results, class_names)



image_copy = image.copy()
image_with_overlay = image.copy()
image_with_weighted_overlay = image.copy()
stat_circles = []
for ball_box in ball_boxes:
    b = 10
    x_min, y_min, x_max, y_max = ball_box
    ball_box = (x_min - b, y_min - b, x_max + b, y_max + b)

    #ball_mask = create_ball_mask_with_kmeans(image_copy, ball_box)
    #ball_mask = create_ball_mask_2(image_copy, ball_box)
    ball_mask, circles = create_ball_mask_3(image_copy, ball_box)

    stat_circles.extend(circles)

    cv2.imshow("ball_mask", ball_mask)

    image_with_overlay = create_ball_overlay(image_with_overlay, ball_mask, ball_box)
    cv2.imshow("image_with_overlay", image_with_overlay)

    image_with_overlay2 = add_bounding_box(image_with_overlay, ball_box, "ball")
    cv2.imshow("image_with_overlay2", image_with_overlay2)

    # add mask weighted
    image_with_weighted_overlay = create_ball_overlay(image_with_weighted_overlay, ball_mask, ball_box, 0.5)
    cv2.imshow("image_with_weighted_overlay", image_with_weighted_overlay)


create_statistical_centroid_analysis(ball_boxes, stat_circles)


# Show images
#resized_image = scale_image(image_bounding_box, 0.8)
cv2.imshow("Detected Objects", image_bounding_box)
#cv2.imwrite("./output/detected_ball.jpg", image_bounding_box)
cv2.waitKey(0)
cv2.destroyAllWindows()
