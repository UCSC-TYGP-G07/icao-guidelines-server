from fastapi import status, FastAPI, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

from blur.laplacian import laplacian
from varied_background.grab_cut_mean import check_varied_bg
from geometric_tests.geometric_tests import valid_geometric

from PIL import Image
import uuid
import os

if not os.path.isdir("./images/"):
    os.makedirs("./images/")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend domain or restrict it to specific domains
    allow_methods=["POST"],
    allow_headers=["*"],
)


def validate_image(file: UploadFile):
    print("Request received")
    print(f"{file.size} in bytes")

    # Accepting only valid file types
    valid_types = ["jpg", "png", "jpeg", "webp"]
    valid_types.extend([x.upper() for x in valid_types])

    file_type = file.filename.split('.')[-1]

    if file_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Given file are not in the supported formats. Supported formats are {str(valid_types)}"
        )
    if file.size > 5 * 1000 * 1000:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size is too large. Max file size is 10MB per request"
        )

    image_name = str(uuid.uuid4()) + '.' + file_type
    image_path = f"./images/{image_name}"

    with Image.open(file.file) as image:
        # Convert image to RGB color space
        image = image.convert('RGB')

        # Save the image in a standard format (e.g., JPEG)
        with open(image_path, 'wb') as buffer:
            image.save(buffer, 'JPEG')

    return image_path


@app.post("/validate_blurring")
async def validate_blurring(file: UploadFile):
    image_path = validate_image(file)
    is_blurred, laplacian_variance  = laplacian.laplacian_filter(image_path)

    return {
        "is_passed": not is_blurred,
        "laplacian_variance": laplacian_variance  # lower means blurred, higher means sharp
    }


@app.post("/validate_varied_bg")
async def validate_varied_bg(file: UploadFile):
    image_path = validate_image(file)
    is_varied_bg, variance_percentage = check_varied_bg(image_path)

    return {
        "is_passed": not is_varied_bg,
        "bg_variance_percentage": variance_percentage  # lower means varied bg, higher means plain bg
    }


@app.post("/validate_geometry")
async def validate_geometry(file: UploadFile):
    image_path = validate_image(file)
    is_valid, tests = valid_geometric(image_path)

    return {
        "is_passed": is_valid,
        "passed_tests": tests
    }


@app.post("/validate_icao")
async def validate_icao(file: UploadFile):
    image_path = validate_image(file)
    is_varied_bg, variance_percentage = check_varied_bg(image_path)
    is_blurred, var = laplacian.laplacian_filter(image_path)
    is_valid_geometric = valid_geometric(image_path)

    return {
        "image_filename": file.filename,
        "is_icao_compliant": not is_varied_bg and not is_blurred,
        "tests": {
            "blur": {
                "is_blur": is_blurred,
                "var": var
            },
            "varied_bg": {
                "is_varied_bg": is_varied_bg,
                "bg_variance_percentage": variance_percentage
            },
            "valid_geometric": is_valid_geometric
        }
        # inside tests, put the results of the tests by calling each function

    }

