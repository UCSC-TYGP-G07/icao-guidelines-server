import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Define the custom preprocessing function to crop images to a square ratio
# Crop a square from the center of the image
def crop_to_square(image):
    width, height = image.shape[1], image.shape[0]
    new_width = new_height = min(width, height)
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    # Crop the array by selecting a subset of rows and columns
    image = image[int(top):int(bottom), int(left):int(right)]
    return image


def crop_to_square_for_img(image):
    width, height = image.size
    new_width = new_height = min(width, height)
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    # Crop the image file
    image = image.crop((left, top, right, bottom))

    return image


def crop_to_square_top_for_img(image):
    width, height = image.size
    new_size = min(width, height)
    left = 0
    top = 0
    right = new_size
    bottom = new_size

    # Crop the image file
    image = image.crop((left, top, right, bottom))

    return image


def detect_hat_or_cap(image_path):
    # Load model from keras file
    model = tf.keras.models.load_model('hat_classifier_2_97_new.h5')

    # Predict on a single image
    # Load the image, crop to a square and resize
    image = tf.keras.preprocessing.image.load_img(image_path)
    image = crop_to_square_top_for_img(image)
    image = image.resize((178, 178))

    # Convert the image to a numpy array
    input_arr = tf.keras.preprocessing.image.img_to_array(image)

    # Add another dimension (since Keras expects a list of images)
    input_arr = np.array([input_arr])

    # Make predictions
    predictions = model.predict(input_arr)
    predicted_labels = np.round(predictions).flatten()  # Assuming a sigmoid activation in the output layer

    if predicted_labels[0] == 1.0:
        return True  # The person is wearing a hat
    else:
        return False  # The person is not wearing a hat
