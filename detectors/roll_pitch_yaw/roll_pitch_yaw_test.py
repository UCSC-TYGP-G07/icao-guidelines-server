import math

# Get roll angle
def get_roll(face_landmarks):
    left_eye_center_x = (face_landmarks[133].x - face_landmarks[33].x)/2
    left_eye_center_y = (face_landmarks[133].y - face_landmarks[33].y)/2
    right_eye_center_x = (face_landmarks[362].x - face_landmarks[263].x)/2
    right_eye_center_y = (face_landmarks[362].y - face_landmarks[263].y)/2
    dx = abs(right_eye_center_x - left_eye_center_x)
    dy = abs(right_eye_center_y - left_eye_center_y)

    # Calculate roll in degrees
    roll = abs(math.atan2(dy, dx)) / math.pi * 180
    return roll

# Get pitch angle
def get_pitch(face_landmarks):
    nose_y = face_landmarks[1].y
    eye_y = (face_landmarks[33].y + face_landmarks[263].y)/2
    mouth_y = face_landmarks[0].y
    
    pitch_measure = abs(eye_y - nose_y) / abs(mouth_y - nose_y)
    # pd => 3.0 => 0, 7.5 => 45
    # pu => 0.9 => 0, 1.7 => 45
    pitch_down = max(pitch_measure - 3.0, 0) / 4.5 * 45
    pitch_up = max(1 / pitch_measure - 0.9, 0) / 0.8 * 45

    return max(pitch_down, pitch_up) 


# Get yaw angle
def get_yaw(face_landmarks):
    left_dist = abs(face_landmarks[130].x - face_landmarks[127].x)
    right_dist = abs(face_landmarks[359].x - face_landmarks[356].x)
    face_width = abs(face_landmarks[356].x - face_landmarks[127].x)
    return (abs(right_dist - left_dist)/face_width)/0.05 * 8


def is_valid_roll_pitch_yaw(face_landmarks):
    roll = get_roll(face_landmarks)
    yaw = get_yaw(face_landmarks)
    pitch = get_pitch(face_landmarks)
    roll_pitch_yaw_avg = (roll + yaw + pitch) / 3
    is_valid = roll_pitch_yaw_avg < 8
    return is_valid, roll_pitch_yaw_avg
