import cv2
import numpy as np


def extract_hair_by_color(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for hair color in HSV space
    # This is just an example for dark hair; you'd adjust these values based on the desired hair color
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([180, 255, 80])

    # Create a mask using color thresholding
    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

    # Extract the largest contiguous region
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # exclude background
    largest_region_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    return largest_region_mask


def corners_to_mask(shape, corners):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(corners, dtype=np.int32), 255)
    return mask


def check_eye_hair_overlap(image_path, left_eye, right_eye):
    # Extract hair using the earlier function
    hair_mask = extract_hair_by_color(image_path)

    # Convert the iris regions into binary masks
    h, w = hair_mask.shape
    left_eye_mask = corners_to_mask((h, w), left_eye)
    right_eye_mask = corners_to_mask((h, w), right_eye)

    # Check overlap between iris masks and hair mask
    left_overlap = np.any(np.logical_and(hair_mask, left_eye_mask))
    right_overlap = np.any(np.logical_and(hair_mask, right_eye_mask))

    return left_overlap, right_overlap
