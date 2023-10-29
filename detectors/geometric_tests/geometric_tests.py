import os

dirname = os.path.dirname(__file__)

MY_THRESHOLD = (0.3, 0.5)
MX_THRESHOLD = (0.45, 0.55)
# CC_THRESHOLD = (0.5, 0.75)
CC_THRESHOLD = (0.45, 0.7)
# DD_THRESHOLD = (0.6, 0.9)
DD_THRESHOLD = (0.35, 0.65)


# Function to check the vertical position
def check_vertical_position(face_landmarks):
    My = (face_landmarks[168].y + face_landmarks[6].y) / 2
    return MY_THRESHOLD[0] <= My <= MY_THRESHOLD[1]


# Function to check the horizontal position
def check_horizontal_position(face_landmarks):
    Mx = (face_landmarks[234].x + face_landmarks[454].x) / 2
    return MX_THRESHOLD[0] <= Mx <= MX_THRESHOLD[1]


# Function to check the head image width ratio
def check_head_image_width_ratio(face_landmarks):
    dy = face_landmarks[454].y - face_landmarks[234].y
    dx = face_landmarks[454].x - face_landmarks[234].x
    CC = ((dy ** 2) + (dx ** 2)) ** 0.5
    return CC_THRESHOLD[0] <= CC <= CC_THRESHOLD[1]


# Function to check the head image height ratio
def check_head_image_height_ratio(face_landmarks):
    dy = face_landmarks[152].y - face_landmarks[10].y
    dx = face_landmarks[152].x - face_landmarks[10].x
    DD = ((dy ** 2) + (dx ** 2)) ** 0.5
    return DD_THRESHOLD[0] <= DD <= DD_THRESHOLD[1]


def valid_geometric(image_path, face_landmarks):
    # Geometric test - Vertical position
    vp_check = check_vertical_position(face_landmarks)
    # Geometric test - Horizontal position
    hp_check = check_horizontal_position(face_landmarks)
    # Geometric test - Head image width ratio
    hiwr_check = check_head_image_width_ratio(face_landmarks)
    # Geometric test - Head image height ratio
    hihr_check = check_head_image_height_ratio(face_landmarks)

    is_valid = vp_check and hp_check and hiwr_check and hihr_check

    tests = {
        "vertical_position": vp_check,
        "horizontal_position": hp_check,
        "head_image_width_ratio": hiwr_check,
        "head_image_height_ratio": hihr_check
    }

    return is_valid, tests
