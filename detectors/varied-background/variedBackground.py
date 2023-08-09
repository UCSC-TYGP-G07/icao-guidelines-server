import cv2
import numpy as np

def has_plain_background(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding to binarize the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Apply morphological opening to remove small objects from the background
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Apply distance transform to find the distance to the nearest background pixel for each foreground pixel
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # Apply thresholding to create a mask of the background region
    _, bg_mask = cv2.threshold(dist_transform, 0.1*dist_transform.max(), 255, cv2.THRESH_BINARY)

    # Calculate the color histogram of the background region
    bg_pixels = img[bg_mask == 255]
    bg_hist = cv2.calcHist([bg_pixels], [0], None, [256], [0, 256])

    # Calculate the percentage of pixels that belong to the most common color
    max_color_pixels = np.max(bg_hist)
    total_pixels = bg_pixels.shape[0]
    color_percentage = max_color_pixels / total_pixels
    print(color_percentage)

    # Check if the percentage is above the threshold
    threshold = 0.25
    if color_percentage > threshold:
        return True
    else:
        return False


print(has_plain_background("dataset/valid/N230101660.JPG"))