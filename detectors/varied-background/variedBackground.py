import cv2

def has_plain_background(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Calculate the percentage of edge pixels
    height, width = edges.shape
    num_edge_pixels = cv2.countNonZero(edges)
    edge_percentage = num_edge_pixels / (height * width)

    # Check if the percentage is below the threshold
    threshold = 0.05
    if edge_percentage < threshold:
        return True
    else:
        return False

print(has_plain_background("Ravindu_Wegiriya_1.jpg"))