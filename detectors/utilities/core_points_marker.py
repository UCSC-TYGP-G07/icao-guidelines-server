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
    mouth_center = (point_m[0], (face_landmarks[61].y + face_landmarks[291].y) / 2 * image_height)

    return {
        "left_eye_center": left_eye_center,
        "right_eye_center": right_eye_center,
        "point_m": point_m,  # midpoint between the centers of the eyes
        "mouth_center": mouth_center
    }
