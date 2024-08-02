from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel, HttpUrl
from .validate import is_valid_drive_or_photos_url, get_drive_download_url, download_image
import requests
from PIL import Image
from io import BytesIO
import os
import shutil

router = APIRouter()
UPLOAD_DIRECTORY = "./uploaded_images/"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Allowed image extensions and maximum file size (20MB)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB in bytes

class URLRequest(BaseModel):
    url: HttpUrl

def is_valid_image(file_content: bytes) -> bool:
    try:
        image = Image.open(BytesIO(file_content))
        image.verify()  # Verify if it is a valid image
        return image.format in ["JPEG", "PNG"]
    except (IOError, SyntaxError) as e:
        return False

def download_image(url: str) -> bytes:
    response = requests.get(url, stream=True)
    response.raise_for_status()

    if 'image' not in response.headers.get('Content-Type', ''):
        raise HTTPException(status_code=415, detail="URL does not point to an image.")

    file_content = response.content
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File size exceeds 20MB")
    
    return file_content

@router.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    # Check file extension
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not allowed. Only JPEG and PNG are accepted.")
    
    # Check file size
    file_size = await file.read()
    if len(file_size) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds the 20MB limit.")
    
    # Reset the file read pointer
    await file.seek(0)

    # Save the file
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename, "message": "File uploaded successfully"}

@router.post("/upload_image_from_url/")
async def upload_image_from_url(request: URLRequest):
    url_str = str(request.url)
    try:
        # Handle Google Drive URLs
        if is_valid_drive_or_photos_url(url_str):
            if "drive.google.com" in url_str:
                url_str = get_drive_download_url(url_str)
            # Add handling for Google Photos if needed

        # Download and validate the image
        file_content = download_image(url_str)
        if not is_valid_image(file_content):
            raise HTTPException(status_code=400, detail="URL does not point to a valid image.")
        
        # Save the image
        file_name = url_str.split("/")[-1].split("?")[0]  # Remove query parameters if present
        file_location = os.path.join(UPLOAD_DIRECTORY, file_name)
        with open(file_location, "wb") as buffer:
            buffer.write(file_content)

        return {"filename": file_name, "message": "Image downloaded and saved successfully"}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
