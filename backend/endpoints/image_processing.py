from fastapi import FastAPI, APIRouter, HTTPException
from io import BytesIO
from PIL import Image, ImageEnhance
import os, httpx

app = FastAPI()
router = APIRouter()

UPLOAD_DIRECTORY = "./uploaded_images/"
PROCESSED_DIRECTORY = "./processed_images/"
os.makedirs(PROCESSED_DIRECTORY, exist_ok=True)

async def get_chart_type(image_bytes: bytes) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://127.0.0.1:8000/detect_chart/",
            files={"file": ("chart.jpg", BytesIO(image_bytes), "image/jpeg")}
        )
        response.raise_for_status()
        return response.json()

def enhance_image(image: Image.Image) -> Image.Image:
    # Resize image
    max_size = (256, 256)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)

    return image

def improve_quality(image_path: str) -> str:
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
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
    
    with open(processed_path, "rb") as image_file:
        image_bytes = image_file.read()
    
    chart_type = await get_chart_type(image_bytes)

    with open("chart_info.txt", 'w') as file:
        file.write(f"Chart Type: {chart_type}\n")

    return {"processed_image_path": processed_path}

app.include_router(router)
