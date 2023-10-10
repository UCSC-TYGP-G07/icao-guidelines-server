from fastapi import status, FastAPI, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

from icao_photo_validator import ICAOPhotoValidator
import os

if not os.path.isdir("./images/"):
    os.makedirs("./images/")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with frontend domain or restrict it to specific domains
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.post("/validate_icao")
async def validate_icao(file: UploadFile, tests: str = ""):
    if tests == "":
        tests = []
    else:
        tests = tests.split(',')
        tests = [test.strip() for test in tests]

    icao_photo_validator = ICAOPhotoValidator(file, tests=tests)
    return icao_photo_validator.validate()
