from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import io
from PIL import Image

# Initialize the router
router = APIRouter()

# Load the trained model once at the start
model_path = './backend/models/trainedmodelnew.h5'
model = load_model(model_path)

# Mapping from class indices to class names
class_labels = {
    0: "AreaGraph",
    1: "BarGraph",
    2: "BoxPlot",
    3: "BubbleChart",
    4: "FlowChart",
    5: "LineGraph",
    6: "Map",
    7: "NetworkDiagram",
    8: "ParetoChart",
    9: "PieChart",
    10: "ScatterGraph",
    11: "TreeDiagram",
    12: "VennDiagram"
}

# Define the image size expected by the model
img_height, img_width = 224, 224

@router.post("/detect_chart/")
async def detect_chart(file: UploadFile = File(...)):
    try:
        # Read the image file
        image_data = await file.read()
        image_pil = Image.open(io.BytesIO(image_data))
        image_pil = image_pil.convert("RGB")  # Ensure the image is in RGB format
        image_pil = image_pil.resize((img_height, img_width))
        
        # Preprocess the image
        img_array = image.img_to_array(image_pil)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image

        # Predict the class
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_labels[predicted_class_index]
        confidence_score = prediction[0][predicted_class_index]

        return JSONResponse(content={"predicted_class": predicted_class, "confidence_score": float(confidence_score)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

