import cv2
import numpy as np
from kMeansClusting import apply_kmeans
import matplotlib.pyplot as plt

def process_detections(image, yolo_results, yolo_class_names):
    """
    Processes detections and adds bounding boxes to the image.
    It also collects the bounding boxes of detected sports balls.

    Parameters:
    - image (np.ndarray): The image on which the bounding boxes will be drawn.
    - results (list): Detection results (output of the YOLO model).
    - class_names (list): List of class names (e.g., ['person', 'sports ball']).

    Returns:
    - np.ndarray: The image with bounding boxes.
    - list: List of bounding boxes for detected sports balls.
    """
    ball_boxes = []

    for r in yolo_results:
        boxes = r.boxes.xyxy.cpu().numpy()  # Bounding-Box-Koordinaten
        confidences = r.boxes.conf.cpu().numpy()  # Konfidenzwerte
        class_ids = r.boxes.cls.cpu().numpy().astype(int)  # Klassen-IDs

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x_min, y_min, x_max, y_max = map(int, box)
            label = f"{yolo_class_names[class_id]}: {conf:.2f}"

            image = add_bounding_box(image, [x_min, y_min, x_max, y_max], label)

            if yolo_class_names[class_id] == "sports ball":
                ball_boxes.append([x_min, y_min, x_max, y_max])

    return image, ball_boxes


def scale_image(image, scale_factor):
    """
    Resizes an image by a given scale factor.

    Parameters:
    - image (np.ndarray): The input image (BGR format).
    - scale_factor (float): Scaling factor (e.g., 0.5 reduces size by 50%, 2.0 doubles it).

    Returns:
    - np.ndarray: The resized image.
    """
    if scale_factor <= 0:
        raise ValueError("Scale factor must be greater than zero.")

    # Compute new dimensions
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    new_size = (new_width, new_height)

    # Resize the image
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    return resized_image



def get_dynamic_kernel_size(image, factor=0.05, min_size=3, max_size=25):
    """
    Computes a dynamic Gaussian blur kernel size based on image dimensions.

    Parameters:
    - image (np.ndarray): The input image.
    - factor (float): The fraction of the image size to use for kernel size.
    - min_size (int): Minimum kernel size (must be odd).
    - max_size (int): Maximum kernel size (must be odd).

    Returns:
    - (int, int): Kernel size (height, width), both are guaranteed to be odd numbers.
    """
    height, width = image.shape[:2]
    kernel_size = int(min(width, height) * factor)

    # Ensure the kernel size is an odd number
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Clamp kernel size within the allowed range
    kernel_size = max(min_size, min(kernel_size, max_size))

    return (kernel_size, kernel_size)


def create_ball_mask_with_kmeans(image, ball_box, k=3, threshold_value=100):
    """
    Creates a binary mask for the ball in the image using K-Means clustering.

    Parameters:
    - image (np.ndarray): The input image (BGR format).
    - ball_box (tuple): Coordinates of the bounding box [x_min, y_min, x_max, y_max].
    - k (int): Number of clusters for K-Means segmentation (default is 3).
    - threshold_value (int): The threshold value to create a binary mask (default is 100).

    Returns:
    - np.ndarray: The binary mask of the ball.
    """

    x_min, y_min, x_max, y_max = ball_box
    image_ball_box = image[y_min:y_max, x_min:x_max]

    # Apply Gaussian Blur for better K-Means results
    kernel_size = get_dynamic_kernel_size(image_ball_box, 0.1)
    print(kernel_size)
    blurred_image = cv2.GaussianBlur(image_ball_box, kernel_size, 0)

    cv2.imshow("blurred_image", blurred_image)

    # Apply K-Means to segment the image
    image_segmented = apply_kmeans(blurred_image, k, 50)

    cv2.imshow("image_segmented", image_segmented)

    # Convert to grayscale for thresholding
    gray_segmented = cv2.cvtColor(image_segmented, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, ball_mask = cv2.threshold(gray_segmented, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)   # Removes noise
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)  # Closes gaps

    cv2.imshow("ball_mask", ball_mask)
    return ball_mask


def create_ball_mask_2(image, ball_box, k=3, threshold_value=100):
    x_min, y_min, x_max, y_max = ball_box
    image_ball_box = image[y_min:y_max, x_min:x_max].copy()

    img_gray = cv2.cvtColor(image_ball_box, cv2.COLOR_BGR2GRAY)
    kernel_size = (17, 17)
    blurred_image = cv2.GaussianBlur(img_gray, kernel_size, 0)

    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, 1, 20, param1=130, param2=30, minRadius=0, maxRadius=0)
    if circles is not None:
        for x, y, r in circles[0]:
            c = plt.Circle((x, y), r, fill=False, lw=3, ec='C1')
            plt.gca().add_patch(c)
    plt.gcf().set_size_inches((12, 8))
    plt.show()

    # cv2.imshow("blurred_image", blurred_image)
    # circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, 1.2, 100)
    # cv2.imshow("blurred_image_2", blurred_image)
    # ball_mask = cv2.circle(image_ball_box, (50, 50), 30, (0, 100, 100), 3)
    # cv2.imshow("ball_mask", ball_mask)

    return blurred_image



def create_ball_mask_3(image, ball_box):
    x_min, y_min, x_max, y_max = ball_box
    image_ball_box = image[y_min:y_max, x_min:x_max].copy()

    gray = cv2.cvtColor(image_ball_box.copy(), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    equalized = cv2.equalizeHist(blurred)
    cv2.imshow("equalized", equalized)

    adjusted = cv2.convertScaleAbs(equalized, alpha=1.5, beta=30)
    cv2.imshow("adjusted", adjusted)

    gray = cv2.cvtColor(image_ball_box.copy(), cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=300)

    ball_mask = np.zeros_like(image_ball_box)

    detected_circles = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(ball_mask, (x, y), r, (255, 255, 255), -1)
            detected_circles.append((x, y, r))

    return ball_mask, detected_circles


def create_ball_mask(image, ball_box):
    x_min, y_min, x_max, y_max = ball_box
    image_ball_box = image[y_min:y_max, x_min:x_max].copy()

    lower_white = np.array([30, 60, 60])
    upper_white = np.array([255, 255, 255])

    # blur the image for better results
    blur = cv2.blur(image_ball_box, (5, 5))
    # blur0 = cv2.medianBlur(blur, 5)
    # blur1 = cv2.GaussianBlur(blur0, (5, 5), 0)
    # blur2 = cv2.bilateralFilter(blur1, 9, 75, 75)

    cv2.imshow("image_ball_box", image_ball_box)

    mask_ball = cv2.inRange(image_ball_box, lower_white, upper_white)
    cv2.imshow("mask_ball", mask_ball)

    # **DISRUPTION-ERUPTION FILTER (Morphologische Operationen)**
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #mask_closed = cv2.morphologyEx(mask_ball, cv2.MORPH_CLOSE, kernel)  # Schließt Lücken
    #mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)  # Entfernt Rauschen
    #cv2.imshow("mask_opened", mask_opened)

    image_ball_box_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
    image_segmented = apply_kmeans(image_ball_box_rgb, 2)
    cv2.imshow("image_segmented", image_segmented)

    _, ball_mask = cv2.threshold(image_segmented, 100, 255, cv2.THRESH_BINARY)

    return ball_mask





def create_ball_overlay(image, ball_mask, bbox, alpha=1.0):
    """
    Applies a red overlay to the detected ball region in an image.

    Parameters:
    - image (np.ndarray): The original input image (BGR format).
    - ball_mask (np.ndarray): A binary mask (same size as bbox) indicating the ball area.
    - bbox (tuple): The bounding box of the ball in (x_min, y_min, x_max, y_max) format.
    - alpha (float): The transparency of the overlay.

    Returns:
    - np.ndarray: The image with the ball overlay applied.
    """
    x_min, y_min, x_max, y_max = bbox

    # Extract the region of interest (ROI) from the image
    image_ball_box = image[y_min:y_max, x_min:x_max].copy()

    # Create red overlay color for the same region
    overlay_color = np.full_like(image_ball_box, (0, 0, 255), dtype=np.uint8)

    # Apply mask to the overlay
    mask_indices = ball_mask > 0
    image_ball_box[mask_indices] = ((1 - alpha) * image_ball_box[mask_indices] + alpha * overlay_color[mask_indices]).astype(np.uint8)

    # Place modified region back into the original image
    image_with_overlay = image.copy()
    image_with_overlay[y_min:y_max, x_min:x_max] = image_ball_box

    return image_with_overlay


def add_bounding_box(image, bbox, label=""):
    """
    Draws a bounding box with an optional label on an image.

    Parameters:
    - image (np.ndarray): The input image (BGR format).
    - bbox (tuple): Bounding box in (x_min, y_min, x_max, y_max) format.
    - label (str, optional): Text label for the bounding box (default: "").

    Returns:
    - np.ndarray: The image with the bounding box and label drawn.
    """
    x_min, y_min, x_max, y_max = bbox

    # Make a copy to avoid modifying the original image
    image_with_bbox = image.copy()

    # Draw bounding box
    cv2.rectangle(image_with_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Draw label if provided
    if label:
        text_position = (x_min, max(y_min - 10, 10))  # Ensure text is within image bounds
        cv2.putText(image_with_bbox, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image_with_bbox



def create_statistical_centroid_analysis(yolo_boxes, hough_circles):
    distances = []

    for i in range(len(yolo_boxes)):
        x_min, y_min, x_max, y_max = yolo_boxes[i]
        w = x_max - x_min
        h = y_max - y_min
        yolo_centroid = (w / 2, h / 2)

        hough_x, hough_y, _ = hough_circles[i]
        hough_centroid = (float(hough_x), float(hough_y))

        # Berechnung der euklidischen Distanz zwischen den beiden Methoden
        distance = np.linalg.norm(np.array(yolo_centroid) - np.array(hough_centroid))
        distances.append(distance)

        print(f"Bild {i + 1}: YOLO-Centroid {yolo_centroid}, Hough-Centroid {hough_centroid}, Abstand: {distance:.2f}")

    # Statistische Analyse
    mean_error = np.mean(distances)
    std_dev = np.std(distances)
    median_error = np.median(distances)

    print("\nStatistische Analyse der Fehler:")
    print(f"Mittelwert (Mean): {mean_error:.2f} Pixel")
    print(f"Standardabweichung (Std Dev): {std_dev:.2f} Pixel")
    print(f"Median: {median_error:.2f} Pixel")

    # plt.figure(figsize=(8, 5))
    # plt.hist(distances, bins=10, edgecolor='black', alpha=0.7)
    # plt.xlabel("Fehlerdistanz (Pixel)")
    # plt.ylabel("Häufigkeit")
    # plt.title("Histogramm der Abweichungen zwischen YOLO und Hough")
    # plt.show()
