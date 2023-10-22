import os

import cv2


def get_core_face_points(image_path, face_landmarks):
    # Read the input image
    image = cv2.imread(image_path)
    # Get image height and width
    image_height, image_width, _ = image.shape

    # Get the coordinates of the face region
    left_eye_center = ((face_landmarks[362].x + face_landmarks[263].x) / 2 * image_width), (
            (face_landmarks[362].y + face_landmarks[263].y) / 2 * image_height)
    right_eye_center = ((face_landmarks[33].x + face_landmarks[133].x) / 2 * image_width), (
            (face_landmarks[33].y + face_landmarks[133].y) / 2 * image_height)
    point_m = ((left_eye_center[0] + right_eye_center[0]) / 2, (left_eye_center[1] + right_eye_center[1]) / 2)
    # mouth_centre = (point_m[0], (face_landmarks[61].y + face_landmarks[291].y) / 2 * image_height)
    mouth_centre = (face_landmarks[14].x * image_width, face_landmarks[14].y * image_height)

    return {
        "left_eye_center": left_eye_center,
        "right_eye_center": right_eye_center,
        "point_m": point_m,  # midpoint between the centers of the eyes
        "mouth_centre": mouth_centre
    }


def get_face_guidelines(image_path, face_data):
    # Read the input image
    image = cv2.imread(image_path)
    # Get image height and width
    image_height, image_width, _ = image.shape

    core_face_points = face_data["core_points"]
    point_m = core_face_points["point_m"]
    mouth_centre = core_face_points["mouth_centre"]

    face_landmarks = face_data["all_landmarks"]

    # =============== line_v ================
    # Calculate the slope of the line passing through mouth_centre and point_m
    m = (point_m[1] - mouth_centre[1]) / (point_m[0] - mouth_centre[0])

    # Calculate the y-intercept
    b = point_m[1] - m * point_m[0]

    # Calculate the endpoints of the line within the image
    top_x = int(-b / m)  # Calculate x using y = mx + b
    top_endpoint = (top_x, 0)
    bottom_x = int((image_height - b) / m)  # Calculate x using y = mx + b
    bottom_endpoint = (bottom_x, image_height)

    line_v = {"slope": m,
              "intercept": b,
              "top_endpoint": top_endpoint,
              "bottom_endpoint": bottom_endpoint
              }
    # ===================================================

    # =============== right_vertical_line ================
    right_ear = ((face_landmarks[234].x - 2 * (face_landmarks[132].x - face_landmarks[234].x)) * image_width,
                 face_landmarks[234].y * image_height)

    b = right_ear[1] - m * right_ear[0]

    # Calculate the endpoints of the new line within the image
    top_x = int(-b / m)  # Calculate x using y = mx + b
    top_endpoint = (top_x, 0)
    bottom_x = int((image_height - b) / m)  # Calculate x using y = mx + b
    bottom_endpoint = (bottom_x, image_height)

    right_vertical_line = {"slope": m,
                           "intercept": b,
                           "top_endpoint": top_endpoint,
                           "bottom_endpoint": bottom_endpoint
                           }
    # ===================================================

    # =============== left_vertical_line ================
    left_ear = ((face_landmarks[454].x - 2 * (face_landmarks[361].x - face_landmarks[454].x)) * image_width,
                face_landmarks[454].y * image_height)

    b = left_ear[1] - m * left_ear[0]

    # Calculate the endpoints of the new line within the image
    top_x = int(-b / m)  # Calculate x using y = mx + b
    top_endpoint = (top_x, 0)
    bottom_x = int((image_height - b) / m)  # Calculate x using y = mx + b
    bottom_endpoint = (bottom_x, image_height)

    left_vertical_line = {"slope": m,
                          "intercept": b,
                          "top_endpoint": top_endpoint,
                          "bottom_endpoint": bottom_endpoint
                          }
    # ===================================================

    # Draw guidelines on image and save image
    cv2.line(image, (line_v["top_endpoint"][0], 0), (line_v["bottom_endpoint"][0], image_height), (0, 255, 0), 2)
    cv2.line(image, (right_vertical_line["top_endpoint"][0], 0),
             (right_vertical_line["bottom_endpoint"][0], image_height), (0, 255, 0), 2)
    cv2.line(image, (left_vertical_line["top_endpoint"][0], 0),
             (left_vertical_line["bottom_endpoint"][0], image_height), (0, 255, 0), 2)

    guides_image_path = f"./images/guidelines/{image_path.split('/')[-1]}"
    os.makedirs(os.path.dirname(guides_image_path), exist_ok=True)
    cv2.imwrite(guides_image_path, image)

    return {
        "vertical_lines": {
            "line_v": line_v,
            "right": right_vertical_line,
            "left": left_vertical_line
        }
    }
