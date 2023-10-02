from fastapi import status, FastAPI, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

from icao_photo_validator import ICAOPhotoValidator
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


@app.post("/validate_icao")
async def validate_icao(file: UploadFile):
    icao_photo_validator = ICAOPhotoValidator(file)
    return icao_photo_validator.validate()
