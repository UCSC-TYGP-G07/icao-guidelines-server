import cv2
import numpy as np
import sys

sys.path.append('../')
import utilities as ut


def most_common_non_black_pixel(hist):
    non_black_hist = hist[1:]
    most_common_pixel_value = np.argmax(non_black_hist)
    return most_common_pixel_value


def highlight_most_visible_color(background, most_common_pixel_value, variance=10):
    gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    lower_bound = max(0, most_common_pixel_value - variance)
    upper_bound = min(255, most_common_pixel_value + variance)
    mask = cv2.inRange(gray_background, np.array([lower_bound]), np.array([upper_bound]))
    highlighted_image = cv2.bitwise_and(background, background, mask=mask)
    return highlighted_image


def is_plain_colored_background(background, variance=10, threshold=90):
    # Convert the background to grayscale
    gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    # Compute histogram of grayscale values
    hist = cv2.calcHist([gray_background], [0], None, [256], [0, 256])

    # Exclude black pixels from the histogram calculation
    black_pixels = hist[0]
    total_black_pixels = int(black_pixels[0])

    # Find the most common non-black grayscale value
    most_common_pixel_value = most_common_non_black_pixel(hist)

    # Calculate the percentage of the most common pixel value among non-black pixels
    total_non_black_pixels = gray_background.size - total_black_pixels
    most_common_pixel_count = hist[most_common_pixel_value + 1]  # +1 to account for excluding black pixels
    percentage = (most_common_pixel_count / total_non_black_pixels) * 100

    # Highlight the most visible color pixels with variance
    highlighted_image = highlight_most_visible_color(background, most_common_pixel_value, variance)

    # Calculate the percentage of highlighted pixels within variance
    highlighted_gray = cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2GRAY)
    highlighted_pixels = np.count_nonzero(highlighted_gray)
    percentage_within_variance = (highlighted_pixels / total_non_black_pixels) * 100

    # Check if the percentage within variance is above a threshold
    return percentage_within_variance, percentage_within_variance >= threshold  # Adjust the threshold as needed


def get_background(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Create a mask (initialized as background)
    mask = np.zeros(image.shape[:2], np.uint8)

    # Create background and foreground models for GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Define a rectangle around the object to help GrabCut initialize
    rect = (10, 10, image.shape[1] - 10, image.shape[0] - 10)

    # Apply GrabCut algorithm
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask to consider certain regions as probable foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Multiply the mask with the input image to get the foreground
    foreground = image * mask2[:, :, np.newaxis]
    background = image - foreground

    return background


def grab_cut(image_path):
    output_image = get_background(image_path)

    if output_image is not None:
        variance_percentage, is_varied_bg = is_plain_colored_background(output_image)
        data = {
            'variance_percentage': variance_percentage,
            'threshold': 90,
            'is_varied_bg': is_varied_bg,
        }

        ut.logger(
            "grab_cut.csv",
            image_path,
            data
        )
        return is_varied_bg, variance_percentage
    else:
        return False, 0
