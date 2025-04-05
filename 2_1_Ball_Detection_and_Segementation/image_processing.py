import cv2
import numpy as np
from kMeansClusting import apply_kmeans
import matplotlib.pyplot as plt

def process_detections(image, yolo_results, yolo_class_names):
    """
    Processes detections and adds bounding boxes to the image.
    It also collects the bounding boxes of detected sports balls.

    :param image (np.ndarray): The image on which the bounding boxes will be drawn.
    :param results (list): Detection results (output of the YOLO model).
    :param class_names (list): List of class names (e.g., ['person', 'sports ball']).

    :return
        - np.ndarray: The image with bounding boxes.
        - list: List of bounding boxes for detected sports balls.
    """
    ball_boxes = []

    for r in yolo_results:
        boxes = r.boxes.xyxy.cpu().numpy()  # bounding box coordinates
        confidences = r.boxes.conf.cpu().numpy()  # confident values
        class_ids = r.boxes.cls.cpu().numpy().astype(int)  # class id's

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

    :param image (np.ndarray): The input image (BGR format).
    :param scale_factor (float): Scaling factor (e.g., 0.5 reduces size by 50%, 2.0 doubles it).

    :return np.ndarray: The resized image.
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

    :param image (np.ndarray): The input image.
    :param factor (float): The fraction of the image size to use for kernel size.
    :param min_size (int): Minimum kernel size (must be odd).
    :param max_size (int): Maximum kernel size (must be odd).

    :return (int, int): Kernel size (height, width), both are guaranteed to be odd numbers.
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

    :param image (np.ndarray): The input image (BGR format).
    :param ball_box (tuple): Coordinates of the bounding box [x_min, y_min, x_max, y_max].
    :param k (int): Number of clusters for K-Means segmentation (default is 3).
    :param threshold_value (int): The threshold value to create a binary mask (default is 100).

    :return np.ndarray: The binary mask of the ball.
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


def create_ball_mask_with_edge_detection(image, ball_box):
    """
    Detects circles within a specified bounding box in an image using edge detection and Hough Circle Transform.

    :param image (numpy.ndarray): The input image (BGR format).
    :param ball_box (tuple): A tuple (x_min, y_min, x_max, y_max) representing the coordinates of the bounding box.

    :return tuple:
            - ball_mask (numpy.ndarray): A binary mask of the detected circles within the bounding box.
            - detected_circles (list): A list of tuples, where each tuple represents a detected circle
              in the format (x, y, radius).
    """
    # Bounding box coordinates
    x_min, y_min, x_max, y_max = ball_box

    # Crop the region of interest (ROI) from the original image based on the bounding box
    image_ball_box = image[y_min:y_max, x_min:x_max].copy()

    # Convert the cropped image to grayscale for edge detection
    gray = cv2.cvtColor(image_ball_box.copy(), cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve circle detection
    gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Detect circles using Hough Circle Transform
    # - dp: Inverse ratio of resolution
    # - minDist: Minimum distance between detected centers
    # - param1: Upper threshold for Canny edge detector
    # - param2: Threshold for center detection
    # - minRadius and maxRadius: Limits for circle size
    circles = cv2.HoughCircles(
        gray_blurred, cv2.HOUGH_GRADIENT, dp=1.4, minDist=100,
        param1=50, param2=30, minRadius=10, maxRadius=500
    )

    ball_mask = np.zeros_like(image_ball_box)
    detected_circles = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw a filled white circle on the mask at the detected location
            cv2.circle(ball_mask, (x, y), r, (255, 255, 255), -1)
            detected_circles.append((x, y, r))

    return ball_mask, detected_circles


def create_ball_mask_with_color(image, ball_box, threshold_value=120):
    """
    Creates a binary mask identifying regions of the image that correspond to a ball
    based on color thresholding and optional blur.

    :param image (numpy.ndarray): The input image (BGR format).
    :param ball_box (tuple): A tuple (x_min, y_min, x_max, y_max) representing the coordinates of the bounding box
                             around the area of interest (e.g., the ball).
    :param threshold_value (int, optional): The threshold value for binarization after color thresholding (default is 120).

    :return numpy.ndarray: A binary mask where the ball region is white (255) and the rest is black (0).
    """
    x_min, y_min, x_max, y_max = ball_box
    # Crop the image to the specified bounding box region
    image_ball_box = image[y_min:y_max, x_min:x_max].copy()

    # Define the lower and upper bounds for white color detection in the image (RGB format)
    lower_white = np.array([100, 120, 100])
    upper_white = np.array([255, 255, 255])

    # Blur the image to reduce noise and improve results of thresholding
    blur = cv2.blur(image_ball_box, (5, 5))

    # Optionally, other blur techniques
    # blur0 = cv2.medianBlur(blur, 5)
    # blur1 = cv2.GaussianBlur(blur0, (5, 5), 0)
    # blur2 = cv2.bilateralFilter(blur1, 9, 75, 75)

    cv2.imshow("image_ball_box", image_ball_box)

    # Create a mask where pixels in the range [lower_white, upper_white] are white (255) and others are black (0)
    ball_mask = cv2.inRange(blur, lower_white, upper_white)
    _, ball_mask = cv2.threshold(ball_mask, threshold_value, 255, cv2.THRESH_BINARY)

    # Optionally, morphological operations to improve the mask
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # mask_closed = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)  # Closes small gaps in the mask
    # mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)  # Removes noise in the mask
    # cv2.imshow("mask_opened", mask_opened)

    # Return the binary mask
    return ball_mask



def create_ball_overlay(image, ball_mask, bbox, alpha=1.0):
    """
    Applies a red overlay to the detected ball region in an image.

    :param image (np.ndarray): The original input image (BGR format).
    :param ball_mask (np.ndarray): A binary mask (same size as bbox) indicating the ball area.
    :param bbox (tuple): The bounding box of the ball in (x_min, y_min, x_max, y_max) format.
    :param alpha (float): The transparency of the overlay.

    :return np.ndarray: The image with the ball overlay applied.
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

    :param image (np.ndarray): The input image (BGR format).
    :param bbox (tuple): Bounding box in (x_min, y_min, x_max, y_max) format.
    :param label (str, optional): Text label for the bounding box (default: "").

    :return np.ndarray: The image with the bounding box and label drawn.
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


# Array for saving the centroid distances for later statistical analysis
distances_for_statistical_analysis = []

def append_statistical_centroid_analysis(image_name, image, yolo_boxes, hough_circles, extend_bbox_px):
    """
    Appends the Euclidean distance between the centroids of YOLO bounding boxes and Hough circles for statistical analysis.

    This function processes the detected bounding boxes and Hough circles, calculates their centroids,
    computes the Euclidean distance between corresponding centroids, and stores these distances for further analysis.
    It also visualizes the centroids on a cropped region of the image.

    :param image_name (str): The name of the image being processed (used for displaying results).
    :param image (numpy.ndarray): The original image where the bounding boxes and circles are detected.
    :param yolo_boxes (list of tuples): List of YOLO bounding boxes, each represented as (x_min, y_min, x_max, y_max).
    :param hough_circles (list of tuples): List of Hough circles, each represented as (x_center, y_center, radius).
    :param extend_bbox_px (int): The number of pixels to extend the bounding box around the detected region.
    """
    for i in range(len(hough_circles)):
        # Extract YOLO bounding box coordinates
        x_min, y_min, x_max, y_max = yolo_boxes[i]

        # Extend bounding box by the specified number of pixels
        ball_box_wide = (x_min - extend_bbox_px, y_min - extend_bbox_px, x_max + extend_bbox_px, y_max + extend_bbox_px)
        x_min, y_min, x_max, y_max = ball_box_wide

        # Calculate width and height of the bounding box
        w = x_max - x_min
        h = y_max - y_min

        # Calculate centroid of the YOLO bounding box (center of the box)
        yolo_centroid = (w / 2, h / 2)

        # Extract centroid from Hough Circle detection
        hough_x, hough_y, _ = hough_circles[i]
        hough_centroid = (float(hough_x), float(hough_y))

        # Compute Euclidean distance between YOLO and Hough centroids
        distance = np.linalg.norm(np.array(yolo_centroid) - np.array(hough_centroid))
        distances_for_statistical_analysis.append(distance)

        # Print the results for this image
        print(f"{image_name}: YOLO Centroid {yolo_centroid}, Hough Centroid {hough_centroid}, Distance: {distance:.2f}")

        # Draw centroids on a cropped region of the original image
        image_ball_box = image[y_min:y_max, x_min:x_max].copy()
        image_with_centroids = draw_centroids_on_image(image_ball_box, yolo_centroid, hough_centroid)
        cv2.imshow(f"{image_name}_centroids (YOLO = red, Hough = green)", image_with_centroids)


def create_statistical_analysis():
    """
    Performs statistical analysis on the distances between the YOLO and Hough centroids.

    This function calculates the mean, standard deviation, and median of the Euclidean distances between
    the centroids of YOLO bounding boxes and Hough circles, and displays these statistics.
    """
    # --- Statistical Analysis of the distances ---
    mean_error = np.mean(distances_for_statistical_analysis)
    std_dev = np.std(distances_for_statistical_analysis)
    median_error = np.median(distances_for_statistical_analysis)

    # Print overall statistics
    print("\nStatistical analysis of the errors:")
    print(f"Mean: {mean_error:.4f} pixels")
    print(f"Standard deviation: {std_dev:.4f} pixels")
    print(f"Median: {median_error:.4f} pixels")

    # Optional: Histogram plot of error distribution
    # plt.figure(figsize=(8, 5))
    # plt.hist(distances_for_statistical_analysis, bins=5, edgecolor='black', alpha=0.7)
    # plt.xlabel("Error distance (pixels)")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of YOLO vs Hough centroid deviations")
    # plt.show()


def draw_centroids_on_image(image, yolo_centroid, hough_centroid):
    """
    Draws the centroids of YOLO and Hough detections on the image.

    :param image (numpy.ndarray): The image where the centroids will be drawn.
    :param yolo_centroid (tuple): The (x, y) coordinates of the YOLO centroid.
    :param hough_centroid (tuple): The (x, y) coordinates of the Hough centroid.

    :return numpy.ndarray: The image with the centroids drawn on it.

    :note The YOLO centroid is drawn in red, and the Hough centroid is drawn in green.
    """
    image_with_centroids = image.copy()

    # Draw YOLO centroid in red
    cv2.circle(image_with_centroids, (int(yolo_centroid[0]), int(yolo_centroid[1])), 2, (0, 0, 255), -1)

    # Draw Hough centroid in green
    cv2.circle(image_with_centroids, (int(hough_centroid[0]), int(hough_centroid[1])), 2, (0, 255, 0), -1)

    return image_with_centroids

