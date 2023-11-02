import cv2
import numpy as np


def check_light_or_dark_score(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Find the maximum and minimum gray levels before processing
    max_gray_level = np.max(image)
    min_gray_level = np.min(image)

    # Calculate the histogram of the grayscale image
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # # Normalize the histogram
    # hist /= hist.sum()

    # # sort the histogram in ascending order
    sorted_hist = np.sort(hist, axis=None)

    # Discard 1% of the lowest and highest pixels
    percentile = 5
    lower_percentile = np.percentile(hist, percentile)
    upper_percentile = np.percentile(hist, 100 - percentile)

    # Find the indices of the pixels to keep
    lower_indices = np.nonzero(sorted_hist >= lower_percentile)[0]
    upper_indices = np.nonzero(sorted_hist <= upper_percentile)[0]

    # Create a mask to keep only the pixels within the desired range
    mask = (lower_indices[0] <= image) & (image <= upper_indices[-1])
    processed_image = np.where(mask, image, 0)

    # Find the maximum and minimum gray levels after processing
    max_gray_level_processed = upper_indices[-1]
    min_gray_level_processed = lower_indices[0]

    # print(f"Maximum Gray Level Before Processing: {max_gray_level}")
    # print(f"Minimum Gray Level Before Processing: {min_gray_level}")
    #
    # print(f"Maximum Gray Level After Processing: {max_gray_level_processed}")
    # print(f"Minimum Gray Level After Processing: {min_gray_level_processed}")

    # Calculate the grayscale histogram
    # hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten()

    # Define the gray level you want to get the frequency for
    desired_gray_level = min_gray_level_processed  # Change this to the desired gray level

    # Get the frequency of the specified gray level
    frequency = hist[desired_gray_level]

    sum_gray_times_frequency = 0
    for gray_level in range(min_gray_level_processed, max_gray_level_processed + 1):
        sum_gray_times_frequency += gray_level * hist[gray_level]

    sum_frequency = sum(hist[min_gray_level_processed:max_gray_level_processed + 1])

    i1 = sum_gray_times_frequency / sum_frequency
    i2 = (min_gray_level_processed + max_gray_level_processed) / 2
    m = (i1 + i2) / 2

    compliance_score = 100 * (1 - (abs(128 - m) / 128))
    return compliance_score

