import cv2
import numpy as np

def get_background(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Create a mask (initialized as background)
    mask = np.zeros(image.shape[:2], np.uint8)

    # Create background and foreground models for GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Define a rectangle around the object to help GrabCut initialize
    rect = (10, 10, image.shape[1] - 10, image.shape[0] - 10)

    # Apply GrabCut algorithm
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask to consider certain regions as probable foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Multiply the mask with the input image to get the foreground
    foreground = image * mask2[:, :, np.newaxis]
    background = image - foreground

    return background   

if __name__ == "__main__":
    image_path = "dataset/valid/N230101743.JPG"  # Replace with the path to your image
    output_image = get_background(image_path)


    if output_image is not None:
        # Display the original and processed images
        cv2.imshow("Original Image", cv2.imread(image_path))
        cv2.imshow("Background", output_image)

        # Wait for a key press and then close the windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()
