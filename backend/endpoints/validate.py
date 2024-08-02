from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
import requests
from PIL import Image
from io import BytesIO
import re
import os
import time

router = APIRouter()

class URLRequest(BaseModel):
    url: HttpUrl

MAX_SIZE_MB = 20
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5  # Initial delay in seconds
UPLOAD_DIRECTORY = "./uploaded_images/"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

def download_image(url: str) -> bytes:
    retries = 0
    retry_delay = INITIAL_RETRY_DELAY
    while retries < MAX_RETRIES:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()

            if len(response.content) > MAX_SIZE_BYTES:
                raise HTTPException(status_code=413, detail="File size exceeds 20MB")

            return response.content
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                # Handle rate limiting (429 Too Many Requests)
                print("Rate limit hit. Retrying...")
                time.sleep(retry_delay)
                retries += 1
                retry_delay *= 2  # Exponential backoff
            else:
                raise HTTPException(status_code=response.status_code, detail=str(e))
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=str(e))

    raise HTTPException(status_code=429, detail="Rate limit exceeded, please try again later")

def is_valid_image(file_content: bytes) -> bool:
    try:
        image = Image.open(BytesIO(file_content))
        image.verify()  # Verify if it is a valid image
        return image.format in ["JPEG", "PNG"]
    except (IOError, SyntaxError) as e:
        print(f"Invalid image: {e}")
        return False

def is_valid_drive_or_photos_url(url: str) -> bool:
    drive_pattern = re.compile(r"https://drive\.google\.com/.+")
    photos_pattern = re.compile(r"https://photos\.google\.com/.+")
    return bool(drive_pattern.match(url) or photos_pattern.match(url))

def get_drive_download_url(drive_url: str) -> str:
    file_id_match = re.search(r"/d/([a-zA-Z0-9_-]+)", drive_url)
    if not file_id_match:
        raise HTTPException(status_code=400, detail="Invalid Google Drive URL")
    file_id = file_id_match.group(1)
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def save_image(file_content: bytes, filename: str):
    file_location = os.path.join(UPLOAD_DIRECTORY, filename)
    with open(file_location, "wb") as buffer:
        buffer.write(file_content)

@router.post("/validate_url/")
async def validate_url(request: URLRequest):
    url_str = str(request.url)  # Convert HttpUrl to string
    print(f"Validating URL: {url_str}")  # Debug print

    try:
        # Handle Google Drive URLs
        if is_valid_drive_or_photos_url(url_str):
            if "drive.google.com" in url_str:
                url_str = get_drive_download_url(url_str)
            # Add handling for Google Photos if needed

        file_content = download_image(url_str)
        if is_valid_image(file_content):
            filename = "image_from_url.jpg"  # You might want to derive a more unique name or extension
            save_image(file_content, filename)
            return {"url": url_str, "valid": True, "message": "URL is valid and image saved"}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error: {e}")  # Debug print
        raise HTTPException(status_code=400, detail="URL is invalid or does not point to a valid image")

    raise HTTPException(status_code=400, detail="URL is invalid or does not point to a valid image")
