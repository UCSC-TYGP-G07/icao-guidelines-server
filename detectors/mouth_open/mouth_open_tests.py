import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import cv2

dirname = os.path.dirname(os.path.dirname(__file__))


def get_face_landmarks(image_path):
    model_path = os.path.join(dirname, "geometric_tests/face_landmarker.task")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    frame = cv2.imread(image_path)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(image)
    face_landmarks_list = detection_result.face_landmarks
    face_landmarks = face_landmarks_list[0]
    return face_landmarks


def mouth_open_test(face_landmarks):
    dy = face_landmarks[14].y - face_landmarks[13].y
    dx = face_landmarks[14].x - face_landmarks[13].x
    d = ((dy**2) + (dx**2)) ** 0.5
    print(d)
    return d < 0.015


def is_mouth_closed(image_path):
    face_landmarks = get_face_landmarks(image_path)
    mouth_close_check = mouth_open_test(face_landmarks)

    return mouth_close_check
