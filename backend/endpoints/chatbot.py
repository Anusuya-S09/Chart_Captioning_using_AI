import google.generativeai as genai
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv
import csv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-pro-latest') 

router = APIRouter()

def load_csv(file_path):
    """Load the CSV data and format it as a string."""
    csv_content = ""
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        csv_content += ", ".join(headers) + "\n"
        for row in reader:
            csv_content += ", ".join(row) + "\n"
    return csv_content

def load_contextual_background(file_path):
    """Load the contextual background from a CSV file."""
    with open(file_path, 'r') as file:
        return file.read()

def generate_response(prompt):
    """Generate a response based on the provided prompt using the generative model."""
    response = model.generate_content(prompt)
    return response.text

@router.get("/chatbot")
async def chatbot(user_input: str):
    context = load_contextual_background('contextual_background_report.md')
    csv_data = load_csv('extracted_table.csv')
    combined_data = f"Markdown Data:\n{context}\n\nCSV Data:\n{csv_data}"

    print("Welcome! I am your chart analysis assistant.")
    print("Ask me anything about the chart, such as its data, trends, patterns, or applications.")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye! If you have more questions in the future, feel free to ask.")
            break
        
        # Create prompt for the model with additional context
        prompt = (f"Context: {combined_data}\n"
                  f"Purpose: You are an analyst helping the user understand a visual chart by providing detailed contextual information.\n"
                  f"User: {user_input}\n"
                  f"Response:")
        
    # Generating response
    try:
        response = generate_response(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

    return {"response": response}

