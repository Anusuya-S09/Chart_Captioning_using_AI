from fastapi import FastAPI, APIRouter, HTTPException
from PIL import Image, ImageEnhance
import os
import numpy as np

app = FastAPI()
router = APIRouter()

UPLOAD_DIRECTORY = "./uploaded_images/"
PROCESSED_DIRECTORY = "./processed_images/"
os.makedirs(PROCESSED_DIRECTORY, exist_ok=True)

from PIL import Image, ImageEnhance

def enhance_image(image: Image.Image) -> Image.Image:
    # Resize image
    max_size = (800, 800)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)  # Use Image.Resampling.LANCZOS instead of ANTIALIAS
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)  # Increase contrast by 50%

    # Convert to grayscale if needed
    # image = image.convert("L")

    return image


def improve_quality(image_path: str) -> str:
    try:
        image = Image.open(image_path)
        enhanced_image = enhance_image(image)
        processed_path = os.path.join(PROCESSED_DIRECTORY, os.path.basename(image_path))
        enhanced_image.save(processed_path)
        return processed_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.get("/process_image/{filename}")
async def process_image(filename: str):
    image_path = os.path.join(UPLOAD_DIRECTORY, filename)
    if not os.path.isfile(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    processed_path = improve_quality(image_path)
    return {"processed_image_path": processed_path}

app.include_router(router)