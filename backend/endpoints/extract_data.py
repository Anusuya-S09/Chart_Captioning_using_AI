from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import pandas as pd
import torch
import io
from io import BytesIO
import re
import httpx
from typing import Dict

router = APIRouter()

def save_table_as_csv(extracted_table: str, output_file_path: str):
    sections = re.split(r'\s+&&&\s+', extracted_table)
    sections = [section.strip() for section in sections if section.strip()]

    if len(sections) < 2:
        raise ValueError("Extracted table data is insufficient or malformed.")

    header = sections[1].split('|')
    header = [col.strip() for col in header if col.strip()]

    rows = [section.split('|') for section in sections[2:]]
    rows = [[col.strip() for col in row] for row in rows]

    max_columns = len(header)
    rows = [row + [''] * (max_columns - len(row)) for row in rows]

    try:
        df = pd.DataFrame(rows, columns=header)
    except ValueError as e:
        raise ValueError(f"Failed to create DataFrame: {str(e)}")

    df.to_csv(output_file_path, index=False)
    print(f"Table saved to {output_file_path}")

def save_title_and_chart_type(title: str, text_file_path: str):
    with open(text_file_path, 'a') as file:
        file.write(f"Chart Title: {title}\n")
    print(f"Title and Chart Type saved to {text_file_path}")

# Initialize the model and processor globally
model_name = "khhuang/chart-to-table"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = DonutProcessor.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@router.post("/extract")
async def extract_data(file: UploadFile = File(...)):
    # Load image
    image_bytes = await file.read()
    img = Image.open(BytesIO(image_bytes))

    # Prepare inputs
    input_prompt = "<data_table_generation> <s_answer>"
    pixel_values = processor(img.convert("RGB"), random_padding=False, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt", max_length=510).input_ids.to(device)

    # Generate outputs
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.config.decoder.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=4,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True
    )

    # Decode and process output
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    extracted_table = sequence.split("<s_answer>")[1].strip()

    # Debugging: Print the raw extracted table
    print("Extracted Table:", extracted_table)

    # Parse the extracted table
    sections = re.split(r'\s+&&&\s+', extracted_table)
    title = re.split(r'\s+&&&\s+', extracted_table)[0].strip()
    sections = [section.strip() for section in sections if section.strip()]

    # Check for empty sections or malformed data
    if len(sections) < 2:
        return {"error": "Extracted table data is insufficient or malformed."}

    # Determine header and rows
    header = sections[1].split('|')
    header = [col.strip() for col in header if col.strip()]
    
    # Debugging: Print header
    print("Header:", header)

    # Extract rows
    rows = [section.split('|') for section in sections[2:]]
    rows = [[col.strip() for col in row] for row in rows]

    # Ensure all rows have the same number of columns as the header
    max_columns = len(header)
    rows = [row + [''] * (max_columns - len(row)) for row in rows]

    # Debugging: Print rows
    print("Rows:", rows)

    # Create DataFrame
    try:
        df = pd.DataFrame(rows, columns=header)
    except ValueError as e:
        return {"error": f"Failed to create DataFrame: {str(e)}"}

    # Save DataFrame to a CSV file in memory
    output_csv_path = "extracted_table.csv"
    save_table_as_csv(extracted_table, output_csv_path)
        
    text_file_path = "chart_info.txt"
    save_title_and_chart_type(title, text_file_path)
        
    return {"message": "Data extracted and saved successfully"}
