import os
import cv2
import numpy as np

zone_names = ["zone_f", "zone_c", "zone_r", "zone_l"]
channel_indices = {"red": 0, "green": 1, "blue": 2}


def get_measurement_zone_coordinates(image_path, face_data):
    # Read the input image
    image = cv2.imread(image_path)

    face_core_points = face_data["core_points"]
    _IED = abs(face_core_points["left_eye_center"][0] - face_core_points["right_eye_center"][0])  # inter-eye distance
    _EM = abs(face_core_points["mouth_centre"][1] - face_core_points["point_m"][1])  # eye-mouth distance
    _MP = 0.3 * _IED

    center_points = {
        "zone_f": (face_core_points["point_m"][0], face_core_points["point_m"][1] - _EM / 2),
        "zone_c": (face_core_points["point_m"][0], face_core_points["point_m"][1] + 3 * _EM / 2),
        "zone_r": (
            face_core_points["right_eye_center"][0] - _MP / 2, face_core_points["point_m"][1] + (_EM / 2 + _MP / 2)),
        "zone_l": (
            face_core_points["left_eye_center"][0] + _MP / 2, face_core_points["point_m"][1] + (_EM / 2 + _MP / 2))
    }

    # Mark core points
    for key, center_point in face_core_points.items():
        center_x, center_y = int(center_point[0]), int(center_point[1])
        cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)

    # Mark specified center points
    for zone_name, point_coordinates in center_points.items():
        center_x, center_y = int(point_coordinates[0]), int(point_coordinates[1])
        label = zone_name
        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(image, label, (center_x - 20, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Save the image with marked points
    zone_centers_image_path = f"./images/zone_centers/{image_path.split('/')[-1]}"
    os.makedirs(os.path.dirname(zone_centers_image_path), exist_ok=True)
    cv2.imwrite(zone_centers_image_path, image)

    cv2.imwrite(zone_centers_image_path, image)

    coordinates = {}

    # Calculate the pixel coordinates of the four corners for each zone
    for zone_name in zone_names:
        center_x, center_y = center_points[zone_name]
        half_length = _MP / 2
        top_left = (int(center_x - half_length), int(center_y - half_length))
        bottom_right = (int(center_x + half_length), int(center_y + half_length))
        coordinates[zone_name] = {"top_left": top_left, "bottom_right": bottom_right}

    return coordinates


def get_illumination_intensity(image_path, zone_coordinates):
    # Read the input image
    image = cv2.imread(image_path)
    # Get image height and width
    image_height, image_width, _ = image.shape

    mean_intensity_values = {"red": {}, "green": {}, "blue": {}}

    # Extract the image in each zone and calculate the mean intensity of each channel
    for zone_name, coordinates in zone_coordinates.items():
        top_left = coordinates['top_left']
        bottom_right = coordinates['bottom_right']
        zone_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Calculate the mean intensity of each channel in the zone
        for channel in channel_indices.keys():
            mean_intensity = np.mean(zone_image[:, :, channel_indices[channel]])  # Use the integer index
            mean_intensity_values[channel][zone_name] = round(mean_intensity, 4)

    return mean_intensity_values


def check_illumination_intensity(image_path, face_data):
    coordinates = get_measurement_zone_coordinates(image_path, face_data)
    mean_intensity_values = get_illumination_intensity(image_path, coordinates)

    is_passed = True

    # Ensure the lowest mean intensity in each channel is not lower than 50% of the highest
    for channel in channel_indices.keys():
        channel_values = mean_intensity_values[channel].values()
        max_intensity = max(channel_values)
        for zone_name, mean_intensity in mean_intensity_values[channel].items():
            if mean_intensity < 0.5 * max_intensity:
                is_passed = False
                break

    # Change the format of the output
    mean_intensity_values_under_zone = {}

    for zone_name in zone_names:
        r_value = mean_intensity_values['red'][zone_name]
        g_value = mean_intensity_values['green'][zone_name]
        b_value = mean_intensity_values['blue'][zone_name]
        mean_intensity_values_under_zone[zone_name] = (r_value, g_value, b_value)

    return is_passed, mean_intensity_values


def check_shadows_across_face(image_path, face_mask_path, face_data):
    # Read the input image
    face_oval_image = cv2.imread(face_mask_path)

    # Get a count of all the pixels that are not fully black
    total_nonzero_pixels = np.count_nonzero(face_data['oval_mask'])

    # Convert the original image to the XYZ color space
    xyz_image = cv2.cvtColor(face_oval_image, cv2.COLOR_BGR2XYZ)

    # Extract the Z channel from the XYZ color space
    z_channel = xyz_image[:, :, 2]

    # Binarize the Z channel to identify shadows (adjust the threshold as needed)
    threshold = 64
    z_bin = cv2.threshold(z_channel, threshold, 255, cv2.THRESH_BINARY)[1]

    # Save the binarized Z channel as a new image
    binarized_z_image_path = f"./images/face_oval_binarized_z/{image_path.split('/')[-1]}"
    os.makedirs(os.path.dirname(binarized_z_image_path), exist_ok=True)
    cv2.imwrite(binarized_z_image_path, z_bin)

    # Calculate the probability of shadows using the percentage of nonzero pixels in the masked ZBin
    nonzero_pixels_in_binarized_z = np.count_nonzero(z_bin)
    probability_of_shadows = (1 - (nonzero_pixels_in_binarized_z / total_nonzero_pixels)) * 100

    # print(nonzero_pixels_in_binarized_z, total_nonzero_pixels)

    # Set threshold for acceptance
    return probability_of_shadows < 25, round(probability_of_shadows, 4)
