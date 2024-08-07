from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Dict
from PIL import Image
import io
import numpy as np
import tensorflow as tf

router = APIRouter()

# Load the trained ResNet50 model
model = tf.keras.models.load_model('./backend/models/resnet50_model.keras')

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Resize image to the target size (assuming 224x224 for ResNet50)
    image = image.resize((224, 224))
    # Convert image to RGB (if not already)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Convert image to numpy array
    image_np = np.array(image)
    # Normalize image array to the range [0, 1]
    image_np = image_np / 255.0
    # Expand dimensions to match the input shape of the model (1, 224, 224, 3)
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

@router.post("/detect_chart/")
async def detect_chart(file: UploadFile = File(...)) -> Dict:
    try:
        # Load the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess the image
        image_np = preprocess_image(image)
        
        # Log the shape of the preprocessed image
        print(f"Preprocessed image shape: {image_np.shape}")
        
        # Make prediction
        predictions = model.predict(image_np)
        
        # Extract the predicted class and confidence
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions).item()  # Convert numpy float to native Python float
        
        # Map the predicted class index to the actual class label
        class_labels = [
            "Area", "Heatmap", "Horizontal Bar", "Horizontal Interval",
            "Line", "Manhattan", "Map", "Pie", "Scatter", "Scatter-Line",
            "Surface", "Venn", "Vertical Bar", "Vertical Box", "Vertical Interval"
        ]
        predicted_class = class_labels[predicted_class_index]
        
        return {"predicted_class": predicted_class, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))