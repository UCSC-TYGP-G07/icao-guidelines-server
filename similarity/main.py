from fastapi import status, FastAPI, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from face_recognition_api import *
import shutil
import uuid
import sys
import os

print("Python version: ", sys.version)
if not os.path.isdir("/images/"):
    os.makedirs("/images/")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend domain or restrict it to specific domains
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/match")
async def match_similarities(file1: UploadFile, file2: UploadFile):
    # 'FACE_RECOGNITION' - Using face recognition algorithm
    CHECKING_METHOD = "FACE_RECOGNITION"

    print("Request received")
    print(f"{file1.size} in bytes")

    # Accepting only valid file types
    valid_types = ["jpg", "png", "jpeg", "webp"]
    valid_types.extend([x.upper() for x in valid_types])

    file1_type = file1.filename.split('.')[-1]
    file2_type = file2.filename.split('.')[-1]

    if (file1_type not in valid_types) or (file2_type not in valid_types):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Given file are not in the supported formats. Supported formats are {str(valid_types)}"
        )
    if (file1.size + file1.size) > 10 * 1000 * 1000:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size is too large. Max file size is 10MB per request"
        )

    image1_name = str(uuid.uuid4()) + '.' + file1_type
    image1_path = f"./images/{image1_name}"

    image2_name = str(uuid.uuid4()) + '.' + file2_type
    image2_path = f"./images/{image2_name}"

    # Saving the files in image directory
    with open(image1_path, "wb") as buffer:
        shutil.copyfileobj(file1.file, buffer)

    with open(image2_path, "wb") as buffer:
        shutil.copyfileobj(file2.file, buffer)

    if CHECKING_METHOD == 'FACE_RECOGNITION':
        try:
            similarity_percentage = face_compare(image1_path, image2_path)
        except FaceNotFound as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        return {"similarity_percentage": similarity_percentage}

    # elif CHECKING_METHOD == 'DLIB':
    #     similarity_percentage = face_distance_dlib(image1_path, image2_path)
    #     return {"similarity_percentage": similarity_percentage}

    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unsupported algorithm")