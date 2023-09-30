# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

dirname = os.path.dirname(__file__)

MY_THRESHOLD = (0.3, 0.5)
MX_THRESHOLD = (0.45, 0.55)
# CC_THRESHOLD = (0.5, 0.75)
CC_THRESHOLD = (0.45, 0.7)
# DD_THRESHOLD = (0.6, 0.9)
DD_THRESHOLD = (0.35, 0.65)


def get_face_landmarks(image_path):
    model_path = os.path.join(dirname, "face_landmarker.task")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)
    face_landmarks_list = detection_result.face_landmarks
    face_landmarks = face_landmarks_list[0]
    return face_landmarks


# Function to check the vertical position
def geometric_test_vertical_position(face_landmarks):
    My = (face_landmarks[168].y + face_landmarks[6].y) / 2
    return MY_THRESHOLD[0] <= My <= MY_THRESHOLD[1]


# Function to check the horizontal position
def geometric_test_horizontal_position(face_landmarks):
    Mx = (face_landmarks[234].x + face_landmarks[454].x) / 2
    return MX_THRESHOLD[0] <= Mx <= MX_THRESHOLD[1]


# Function to check the head image width ratio
def geometric_test_head_image_width_ratio(face_landmarks):
    dy = face_landmarks[454].y - face_landmarks[234].y
    dx = face_landmarks[454].x - face_landmarks[234].x
    CC = ((dy ** 2) + (dx ** 2)) ** 0.5
    return CC_THRESHOLD[0] <= CC <= CC_THRESHOLD[1]


# Function to check the head image height ratio
def geometric_test_head_image_height_ratio(face_landmarks):
    dy = face_landmarks[152].y - face_landmarks[10].y
    dx = face_landmarks[152].x - face_landmarks[10].x
    DD = ((dy ** 2) + (dx ** 2)) ** 0.5
    return DD_THRESHOLD[0] <= DD <= DD_THRESHOLD[1]


def valid_geometric(image_path):
    # image_path = os.path.join(dirname, "Profile-pic.jpeg")
    face_landmarks = get_face_landmarks(image_path)
    # Geometric test - Vertical position
    vp_check = geometric_test_vertical_position(face_landmarks)
    # Geometric test - Horizontal position
    hp_check = geometric_test_horizontal_position(face_landmarks)
    # Geometric test - Head image width ratio
    hiwr_check = geometric_test_head_image_width_ratio(face_landmarks)
    # Geometric test - Head image height ratio
    hihr_check = geometric_test_head_image_height_ratio(face_landmarks)

    data = {
        "valid": vp_check and hp_check and hiwr_check and hihr_check,
        "vertical_position": vp_check,
        "horizontal_position": hp_check,
        "head_image_width_ratio": hiwr_check,
        "head_image_height_ratio": hihr_check
    }

    return data
