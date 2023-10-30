from redeye.redeye import valid_redeye


def check_eyes_open(face_data):
    face_blendshapes = face_data["blendshapes"]
    eye_open_threshold = 0.75

    open_probabilities = {
        "left_eye": round(1 - face_blendshapes[9].score, 4),
        "right_eye": round(1 - face_blendshapes[10].score, 4)
    }

    is_both_open = (open_probabilities["left_eye"] > eye_open_threshold) and (
            open_probabilities["right_eye"] > eye_open_threshold)

    return is_both_open, open_probabilities


def check_looking_away(face_data):
    face_blendshapes = face_data["blendshapes"]

    # Extract eye landmarks
    eye_look_in_left = face_blendshapes[13].score
    eye_look_in_right = face_blendshapes[14].score
    eye_look_out_left = face_blendshapes[15].score
    eye_look_out_right = face_blendshapes[16].score
    eye_look_up = (face_blendshapes[17].score + face_blendshapes[17].score) / 2
    eye_look_down = (face_blendshapes[11].score + face_blendshapes[12].score) / 2

    # Check if looking right
    looking_right = (
            0.3 <= eye_look_in_left <= 0.6 and
            0.0 <= eye_look_in_right <= 0.03 and
            0.0 <= eye_look_out_left <= 0.03 and
            0.2 <= eye_look_out_right <= 0.8
    )

    # Check if looking left
    looking_left = (
            0.0 <= eye_look_in_left <= 0.03 and
            0.3 <= eye_look_in_right <= 0.9 and
            0.7 <= eye_look_out_left <= 1 and
            0.0 <= eye_look_out_right <= 0.03
    )

    # Check if looking up
    looking_up = (
            0.0 <= eye_look_down <= 0.06 and
            0.1 <= eye_look_up <= 0.6
    )

    # Check if looking down
    looking_down = (
            0.17 <= eye_look_down <= 0.5 and
            0.0 <= eye_look_up <= 0.06
    )

    # Check if straight at camera
    straight_at_camera = (
            0.05 <= eye_look_down <= 0.2 and
            0.04 <= eye_look_up <= 0.14 and
            0.0 <= eye_look_in_left <= 0.05 and
            0.05 <= eye_look_in_right <= 0.15 and
            0.0 <= eye_look_out_left <= 0.5 and
            0.02 <= eye_look_out_right <= 0.12

    )

    gaze_directions = {
        "looking_right": looking_right,
        "looking_left": looking_left,
        "looking_up": looking_up,
        "looking_down": looking_down,
    }

    if not (looking_right or looking_left or looking_up or looking_down):
        return straight_at_camera, gaze_directions

    return False, gaze_directions


def check_redeye(image_path, left_iris_coords, right_iris_coords):
    return valid_redeye(image_path, left_iris_coords, right_iris_coords)
