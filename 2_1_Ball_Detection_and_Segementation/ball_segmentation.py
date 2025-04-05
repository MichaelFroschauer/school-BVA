import os
import pickle
import cv2
from ultralytics import YOLO
from image_processing import *
from pathlib import Path


# Load YOLO model
model = YOLO("yolo11n.pt")

# Load image
image_folder = Path("./img")

image_paths = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png")) + list(image_folder.glob("*.jpeg"))

for image_path in image_paths:
    filename = image_path.stem
    image_path = str(image_path)
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
    image_with_segmentation_overlay = image.copy()
    image_with_segmentation_and_bbox_overlay = image.copy()
    image_with_weighted_overlay = image.copy()
    image_with_centroids = image.copy()
    stat_circles = []
    extend_bbox_px = 10
    for i, ball_box in enumerate(ball_boxes):
        x_min, y_min, x_max, y_max = ball_box
        ball_box_wide = (x_min - extend_bbox_px, y_min - extend_bbox_px, x_max + extend_bbox_px, y_max + extend_bbox_px)

        # Segment ball using desired method
        #ball_mask = create_ball_mask_with_color(image_copy, ball_box_wide)
        #ball_mask = create_ball_mask_with_kmeans(image_copy, ball_box_wide)

        ball_mask, circles = create_ball_mask_with_edge_detection(image_copy, ball_box_wide)
        # Collect detected circles for later analysis
        stat_circles.extend(circles)

        cv2.imshow(f"{filename}_ball_mask_{i}", ball_mask)
        #cv2.imwrite(f"./output/{filename}_ball_mask_{i}.png", ball_mask)

        # Create overlays with the segmentation results and the bounding boxes
        image_with_segmentation_overlay = create_ball_overlay(image_with_segmentation_overlay, ball_mask, ball_box_wide)

        image_with_segmentation_and_bbox_overlay = create_ball_overlay(image_with_segmentation_and_bbox_overlay, ball_mask, ball_box_wide)
        image_with_segmentation_and_bbox_overlay = add_bounding_box(image_with_segmentation_and_bbox_overlay, ball_box, "ball")

        image_with_weighted_overlay = create_ball_overlay(image_with_weighted_overlay, ball_mask, ball_box_wide, 0.5)


    # Calculate centroids and distances
    append_statistical_centroid_analysis(filename, image_copy, ball_boxes, stat_circles, extend_bbox_px)

    # Show result images
    cv2.imshow(f"{filename}_image_with_overlay", image_with_segmentation_overlay)
    cv2.imshow(f"{filename}_image_with_segmentation_and_bbox_overlay", image_with_segmentation_and_bbox_overlay)
    cv2.imshow(f"{filename}_image_with_weighted_overlay", image_with_weighted_overlay)
    cv2.imshow(f"{filename}_image_detected_objects", image_bounding_box)

    # cv2.imwrite(f"./output/{filename}_image_with_overlay.png", image_with_segmentation_overlay)
    # cv2.imwrite(f"./output/{filename}_image_with_segmentation_and_bbox_overlay.png", image_with_segmentation_and_bbox_overlay)
    # cv2.imwrite(f"./output/{filename}_image_with_weighted_overlay.png", image_with_weighted_overlay)
    # cv2.imwrite(f"./output/{filename}_image_detected_objects.png", image_bounding_box)


create_statistical_analysis()

cv2.waitKey(0)
cv2.destroyAllWindows()
