import time

from fastapi import status, FastAPI, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

from blur.laplacian import laplacian
from varied_background.grab_cut_mean import grab_cut
from geometric_tests.geometric_tests import valid_geometric
from utilities.mp_face import get_num_faces, get_face_landmarks, get_mp_face_region

from PIL import Image
import uuid
import os


class ICAOPhotoValidator:
    def __init__(self, file: UploadFile, tests: list = None):
        self.file = file
        self.data = {}  # Stores the data such as face_landmarks, etc.
        self.pipeline = {}  # Stores the results of the tests
        self.paths = {}  # Stores the paths of the images (original_image, resized_image)
        self.tests = tests

    # Functions for preprocessing the input image
    def _validate_file(self):
        print("Input file received")
        print(f"{self.file.size} in bytes")

        # Accepting only valid file types
        valid_types = ["jpg", "png", "jpeg", "webp"]
        valid_types.extend([x.upper() for x in valid_types])

        file_type = self.file.filename.split('.')[-1]

        if file_type not in valid_types:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Input file type is not supported, supported image formats are {str(valid_types)}."
            )
        if self.file.size > 5 * 1000 * 1000:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Input file size is too large, max file size allowed is 5MB."
            )

        image_name = str(uuid.uuid4()) + '.' + file_type
        destination_path = f"./images/original/{image_name}"

        # Ensure that the directory structure exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        with Image.open(self.file.file) as image:
            # Convert image to RGB color space
            image = image.convert('RGB')
            # Remove metadata from the image
            # image = Image.frombytes(image.mode, image.size, image.tobytes())

            # Save the image in a standard format (e.g., JPEG)
            with open(destination_path, 'wb') as buffer:
                image.save(buffer, 'JPEG')

        self.paths["original_image"] = destination_path

    def _resize_image(self):
        max_resized_width = 320

        image_name = os.path.basename(self.paths["original_image"])
        destination_path = f"./images/resized/{image_name}"

        # Ensure that the directory structure exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        with Image.open(self.paths["original_image"]) as image:
            # get original image width and height
            width, height = image.size

            resized_image = image.resize((max_resized_width, int(max_resized_width * height / width)))
            resized_image.save(destination_path, 'JPEG')

        self.paths["resized_image"] = destination_path

    def _detect_face(self):
        num_faces = get_num_faces(self.paths["original_image"])
        if num_faces == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in the image."
            )
        elif num_faces > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="More than one face detected in the image."
            )

        self.data["num_faces"] = num_faces

    def _get_face_landmarks(self):
        face_landmarks = get_face_landmarks(self.paths["original_image"])
        self.data.setdefault("face", {}).update({"all_landmarks": face_landmarks})

    def _get_face_region(self):
        mp_face_region = get_mp_face_region(self.paths["original_image"], self.data["face"]["all_landmarks"])
        self.data.setdefault("face", {}).update({"mp_region_coords": mp_face_region})

    # Functions for running the tests
    def _validate_blurring(self):
        is_blurred, blur_var = laplacian.laplacian_filter(self.paths["original_image"])
        return {"is_passed": not is_blurred, "laplacian_variance_value": blur_var}

    def _validate_varied_bg(self):
        is_varied_bg, bg_var = grab_cut(self.paths["resized_image"], self.data["face"]["all_landmarks"])
        return {"is_passed": not is_varied_bg, "bg_variance_percentage": bg_var}

    def _validate_geometry(self):
        is_valid_geometric, geometric_tests = valid_geometric(self.paths["original_image"],
                                                              self.data["face"]["all_landmarks"])
        return {"is_passed": is_valid_geometric, "geometric_tests_passed": geometric_tests}

    def validate(self):
        print("Running ICAO photo validation pipeline")
        # Mapping of test names to corresponding validation methods
        validation_methods = {
            "geometry": self._validate_geometry,  # ICAO-4, ICAO-5, ICAO-6, ICAO-7
            "blurring": self._validate_blurring,  # ICAO-8
            "varied_bg": self._validate_varied_bg,  # ICAO-17
        }

        # Pre-process the input file before running the tests
        self._validate_file()
        self._resize_image()
        self._detect_face()
        self._get_face_landmarks()
        self._get_face_region()

        self.pipeline["all_passed"] = True

        # If tests list is None, run all tests
        tests_to_run = self.tests if self.tests else validation_methods.keys()

        for test_name in tests_to_run:
            validation_method = validation_methods.get(test_name)
            if validation_method:
                try:
                    start = time.time()
                    result = validation_method()  # Run the validation method, which returns a dict
                    end = time.time()
                    result["time_elapsed"] = round(end - start, 3)
                    self.pipeline.setdefault('tests', {}).setdefault(test_name, {}).update(result)  # Update the
                    # result in the pipeline

                    if result["is_passed"] is False:
                        self.pipeline["all_passed"] = False

                        # Uncomment the following line to stop the testing and return if a single test fails
                        # return self.pipeline

                except Exception as e:
                    self.pipeline["all_passed"] = False
                    error_message = f"Error during {test_name} validation: {str(e)}"
                    self.pipeline.setdefault("errors", []).append(error_message)

                    # Uncomment the following line to stop the testing and return if a single test causes an error
                    # return self.pipeline
            else:
                return {"error": f"Invalid test name: {test_name}"}

        # If all tests passed, set "all_passed" to True
        self.pipeline["all_passed"] = self.pipeline.get("all_passed", True)

        return self.pipeline
