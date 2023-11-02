import cv2
import numpy as np
from PIL import Image


def is_washed_out(image_path):
    image = Image.open(image_path)
    image = np.array(image)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    washed_out_score = (np.max(gray_image) - np.min(gray_image)) / 255 * 100

    if washed_out_score < 80 and np.mean(gray_image) > 170:
        return True, washed_out_score
    else:
        return False, washed_out_score
