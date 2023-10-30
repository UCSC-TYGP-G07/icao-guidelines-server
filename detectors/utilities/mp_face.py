import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


def get_num_faces(image_path):
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = face_detection.process(image_rgb)

    # Get the number of faces detected
    num_faces = 0
    if results.detections:
        num_faces = len(results.detections)

    return num_faces


def get_face_landmarks_and_blendshapes(image_path):
    image_name = os.path.basename(image_path)

    # Create an FaceLandmarker object
    base_options = python.BaseOptions(model_asset_path='utilities/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    # Read the input image
    image = mp.Image.create_from_file(image_path)

    # Detect face landmarks from the input image
    detection_result = detector.detect(image)

    # Visualize detection result
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    # Save the annotated image
    image_name = os.path.basename(image_path)
    destination_path = f"./images/landmarks/{image_name}"

    # Ensure that the directory structure exists
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    cv2.imwrite(destination_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    return detection_result.face_landmarks[0], detection_result.face_blendshapes[0]


def get_mp_face_region(image_path, face_landmarks):
    # Read the input image
    image = cv2.imread(image_path)
    # Get image height and width
    image_height, image_width, _ = image.shape

    # Get the coordinates of the face region
    mp_left = face_landmarks[234].x * image_width
    mp_right = face_landmarks[454].x * image_width
    mp_top = face_landmarks[10].y * image_height
    mp_bottom = face_landmarks[152].y * image_height

    return [(mp_left, mp_top), (mp_right, mp_top), (mp_right, mp_bottom), (mp_left, mp_bottom)]


def get_face_oval_mask(image_path, face_landmarks):
    # Read the input image
    image = cv2.imread(image_path)

    # Get image height and width
    image_height, image_width, _ = image.shape

    mp_face_mesh = mp.solutions.face_mesh
    face_oval = mp_face_mesh.FACEMESH_FACE_OVAL

    df = pd.DataFrame(list(face_oval), columns=["p1", "p2"])

    # Ordering face oval lines
    routes_idx = []

    p1 = df.iloc[0]["p1"]
    p2 = df.iloc[0]["p2"]

    for i in range(0, df.shape[0]):
        # print(p1, p2)

        obj = df[df["p1"] == p2]
        p1 = obj["p1"].values[0]
        p2 = obj["p2"].values[0]

        route_idx = []
        route_idx.append(p1)
        route_idx.append(p2)
        routes_idx.append(route_idx)

    # Finding the coordinates of points
    routes = []

    for source_idx, target_idx in routes_idx:
        source = face_landmarks[source_idx]
        target = face_landmarks[target_idx]

        relative_source = (int(image_width * source.x), int(image_height * source.y))
        relative_target = (int(image_width * target.x), int(image_height * target.y))

        # cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness = 2)

        routes.append(relative_source)
        routes.append(relative_target)

    # Extracting face_oval
    mask = np.zeros((image_height, image_width))
    mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
    mask = mask.astype(bool)

    return mask


def extract_face_oval_image(image_path, mask):
    # Read the input image
    image = cv2.imread(image_path)

    # Get image height and width
    image_height, image_width, _ = image.shape

    out = np.zeros_like(image)
    out[mask] = image[mask]

    # Save the face_oval image
    image_name = os.path.basename(image_path)
    face_oval_destination_path = f"./images/face_oval/{image_name}"

    # Ensure that the directory structure exists
    os.makedirs(os.path.dirname(face_oval_destination_path), exist_ok=True)
    cv2.imwrite(face_oval_destination_path, cv2.cvtColor(out[:, :, ::-1], cv2.COLOR_RGB2BGR))
    return face_oval_destination_path


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()
