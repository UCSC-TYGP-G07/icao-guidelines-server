import face_recognition as fr
import cv2
import pandas as pd
from datetime import datetime
import time


class FaceNotFound(Exception):
    def __init__(self, message="No faces detected in one or both images."):
        self.message = message
        super().__init__(self.message)


def logger(similarity_percentage, time, image1, image2):
    time_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    df_headers = pd.DataFrame({
        'log_time': [time_str],
        'similarity_percentage': [similarity_percentage],
        'time_taken': [time],
        'image1': [image1],
        'image2': [image2]
    })
    df_headers.to_csv('logs.csv', mode='a', index=False, header=False)


def face_compare_(path1, path2):
    # Load the images
    image1 = fr.load_image_file(path1)
    image2 = fr.load_image_file(path2)

    image1 = cv2.resize(image1, (0, 0), fx=0.5, fy=0.5)
    image2 = cv2.resize(image2, (0, 0), fx=0.5, fy=0.5)

    # Extract face encodings
    encodings1 = fr.face_encodings(image1)
    encodings2 = fr.face_encodings(image2)

    if len(encodings1) == 0 or len(encodings2) == 0:
        # No faces detected in one or both images
        raise FaceNotFound

    # Calculate face distance
    face_distance = fr.face_distance(encodings1, encodings2)

    # Calculate similarity percentage
    similarity_percentage = (1 - face_distance[0]) * 100
    return similarity_percentage


def face_compare(path1, path2):
    start = time.time()
    similarity_percentage = face_compare_(path1, path2)
    end = time.time()
    logger(similarity_percentage, end - start, path1, path2)
    return similarity_percentage
