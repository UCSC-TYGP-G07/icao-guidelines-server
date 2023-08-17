import utilities as ut
import cv2


THRESHOLD = 75
LOGGER_PATH = "./blur/laplacian.csv"


def is_image_blurry(variance, threshold=THRESHOLD):
    if variance < threshold:
        return True
    return False


def laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def laplacian_filter(image_name):
    image = cv2.imread(image_name)
    var = laplacian_variance(image)
    is_blurred = is_image_blurry(var)

    data = {
        'laplace variance': var,
        'threshold': THRESHOLD,
        'is_blurred': is_blurred,
    }

    ut.logger(
        LOGGER_PATH,
        image_name,
        data
    )

    return is_blurred, var
