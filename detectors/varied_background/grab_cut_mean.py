import os
import time

import cv2
import numpy as np
import sys

sys.path.append('../')
# import utilities as ut

SIMILARITY_THRESHOLD = 70  # SIMILARITY COLOR THRESHOLD
VARIANCE = 10


def most_common_non_black_pixel(hist):
    non_black_hist = hist[1:]
    most_common_pixel_value = np.argmax(non_black_hist)
    return most_common_pixel_value


def highlight_most_visible_color(background, most_common_pixel_value, variance=VARIANCE):
    gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    lower_bound = max(0, most_common_pixel_value - variance)
    upper_bound = min(255, most_common_pixel_value + variance)
    mask = cv2.inRange(gray_background, np.array([lower_bound]), np.array([upper_bound]))
    highlighted_image = cv2.bitwise_and(background, background, mask=mask)
    return highlighted_image


def is_varied_background(background, variance=VARIANCE, threshold=SIMILARITY_THRESHOLD):
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
    # A higher percentage_within_variance value means more pixels will be highlighted (plain background)
    # A lower percentage_within_variance value means less pixels will be highlighted (varied background)
    return percentage_within_variance, percentage_within_variance <= threshold  # Adjust the threshold as needed


def get_background(image_path, face_landmarks, init_method='rect'):
    # Load the image
    image = cv2.imread(image_path)

    # Create a mask (initialized as background)
    mask = np.zeros(image.shape[:2], np.uint8)

    # Create background and foreground models for GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    if (init_method == 'rect'):
        # Define a rectangle around the object to help GrabCut initialize
        rect = (10, 10, image.shape[1] - 20, image.shape[0])

        # Apply GrabCut algorithm
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    else:
        image_width = image.shape[1]
        image_height = image.shape[0]

        # Create a mask (initialized as background)
        mask = np.zeros(image.shape[:2], np.uint8) + cv2.GC_PR_BGD

        # Draw circles on mask to represent face
        # Top part of face
        midpoint1 = (int(face_landmarks[151].x * image_width), int(face_landmarks[151].y * image_height))
        radius1 = int(face_landmarks[151].x * image_width) - int(face_landmarks[21].x * image_width)
        cv2.circle(mask, midpoint1, radius1, cv2.GC_FGD, thickness=cv2.FILLED)

        # Bottom part of face
        midpoint2 = (int(face_landmarks[1].x * image_width), int(face_landmarks[1].y * image_height))
        radius2 = int(face_landmarks[1].x * image_width) - int(face_landmarks[205].x * image_width)
        cv2.circle(mask, midpoint2, radius2, cv2.GC_FGD, thickness=cv2.FILLED)

        # Triangular region to show body
        body_mask = [(0, image_height),
                     (int(face_landmarks[152].x * image_width), int(face_landmarks[152].y * image_height)),
                     (image_width, image_height)]

        # Add the body mask
        cv2.drawContours(mask, [np.array(body_mask)], 0, cv2.GC_FGD, thickness=cv2.FILLED)

        # Apply GrabCut algorithm
        cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

        # save an image with the mask applied on the original image

    # Modify the mask to consider certain regions as probable foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Multiply the mask with the input image to get the foreground
    foreground = image * mask2[:, :, np.newaxis]
    background = image - foreground

    image_name = os.path.basename(image_path)
    destination_path = f"./images/separated-bg/{image_name}"

    # Ensure that the directory structure exists
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    # save background as new image
    cv2.imwrite(destination_path, background)

    return background


def grab_cut(image_path, all_landmarks):
    background = get_background(image_path, all_landmarks)

    if background is not None:
        variance_percentage, is_varied_bg = is_varied_background(background)
        round(variance_percentage, 3)
        data = {
            'variance_percentage': variance_percentage,
            'threshold': SIMILARITY_THRESHOLD,
            'is_varied_bg': is_varied_bg,
        }

        # ut.logger(
        #     "grab_cut.csv",
        #     image_path,
        #     data
        # )
        return is_varied_bg, variance_percentage
    else:
        return False, 0


if __name__ == '__main__':
    image_path = sys.argv[1]
    start = time.time()
    is_varied_bg, variance_percentage = grab_cut(image_path)
    end = time.time()
    print(f"Time elapsed: {end - start}")
    print(f"Is varied background: {is_varied_bg}")
    print(f"Variance percentage: {variance_percentage}")
