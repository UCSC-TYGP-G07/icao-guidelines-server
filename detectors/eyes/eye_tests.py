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
