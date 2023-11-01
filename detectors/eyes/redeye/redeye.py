import cv2

COMPLIANCE_THRESHOLD = 98
RGB_THRESHOLD = (100, 60, 60)


def crop_iris_region(image_path, coords, padding=0):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cropped_image = image[coords[0][1] - padding:coords[2][1] + padding, coords[0][0] - padding:coords[2][0] + padding]
    return cropped_image


def classify_pixels(iris, threshold):
    red_channel = iris[:, :, 0]
    green_channel = iris[:, :, 1]
    blue_channel = iris[:, :, 2]

    # Check which pixels are red (based on threshold)
    red_pixels = (red_channel > threshold[0]) & (green_channel < threshold[1]) & (blue_channel < threshold[2])
    natural_pixels = ~red_pixels

    return red_pixels.sum(), natural_pixels.sum()


def compliance_score(iris, threshold=RGB_THRESHOLD):
    red_count, natural_count = classify_pixels(iris, threshold)
    total_pixels = red_count + natural_count
    if total_pixels == 0:  # avoid division by zero
        return 0

    return (natural_count / total_pixels) * 100


def valid_redeye(image_path, left_iris_coords, right_iris_coords):
    # Get the iris region
    left_iris_image = crop_iris_region(image_path, left_iris_coords, padding=0)
    right_iris_image = crop_iris_region(image_path, right_iris_coords, padding=0)

    # Calculate compliance score
    left_compliance_score = compliance_score(left_iris_image)
    right_compliance_score = compliance_score(right_iris_image)

    avg_compliance_score = (left_compliance_score + right_compliance_score) / 2
    is_redeye = avg_compliance_score < COMPLIANCE_THRESHOLD

    return is_redeye, avg_compliance_score
