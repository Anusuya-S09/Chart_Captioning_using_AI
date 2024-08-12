from fastapi import FastAPI
import uvicorn
from .endpoints.upload import router as upload_router
from .endpoints.validate import router as validate_router
from .endpoints.image_processing import router as image_processing_router
from .endpoints.detect_chart import router as detect_router
from .endpoints.extract_data import router as extract_router

app = FastAPI()

app.include_router(upload_router)
app.include_router(validate_router)
app.include_router(image_processing_router)
app.include_router(detect_router)
app.include_router(extract_router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")