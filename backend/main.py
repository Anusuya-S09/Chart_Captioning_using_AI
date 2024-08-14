from fastapi import FastAPI
import uvicorn
import os,shutil
from fastapi.middleware.cors import CORSMiddleware
from .endpoints.upload import router as upload_router
from .endpoints.validate import router as validate_router
from .endpoints.image_processing import router as image_processing_router
from .endpoints.detect_chart import router as detect_router
from .endpoints.extract_data import router as extract_router
from .endpoints.generate_caption import router as generate_router
from .endpoints.chatbot import router as chatbot_router

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def cleanup_files():
    # Define paths to cleanup
    uploaded_images_path = "uploaded_images/"
    processed_images_path = "processed_images/"
    chart_info_file = "chart_info.txt"
    markdown_file = "contextual_background_report.md"  # Adjust the name as necessary

    # Delete uploaded images
    if os.path.exists(uploaded_images_path):
        shutil.rmtree(uploaded_images_path)
        os.makedirs(uploaded_images_path)  # Recreate the directory if needed

    # Delete processed images
    if os.path.exists(processed_images_path):
        shutil.rmtree(processed_images_path)
        os.makedirs(processed_images_path)  # Recreate the directory if needed

    # Delete chart_info.txt
    if os.path.exists(chart_info_file):
        os.remove(chart_info_file)

    # Delete markdown file
    if os.path.exists(markdown_file):
        os.remove(markdown_file)

app.include_router(upload_router)
app.include_router(validate_router)
app.include_router(image_processing_router)
app.include_router(detect_router)
app.include_router(extract_router)
app.include_router(generate_router)
app.include_router(chatbot_router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    cleanup_files