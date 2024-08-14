# AI Chart Captioning System

## Project Description

The AI Chart Captioning System is a web application designed to automatically generate captions for charts. It consists of a Streamlit frontend for user interaction and a FastAPI backend for processing and generating captions. The system utilizes machine learning models for chart type detection and text recognition, as well as Google's Gemini API for generating contextual information.

## Features

- **Chart Image Upload**: Upload chart images from your local device.
- **Chart URL Validation**: Validate and process chart images from URLs.
- **Chart Type Detection**: Automatically detect the type of chart.
- **Text Extraction and Caption Generation**: Extract text and generate descriptive captions for charts.
- **Chatbot Interaction**: Interactive chatbot for querying and understanding the generated chart captions.

## Requirements

Before running the project, ensure you have the following installed:

- Python 3.7 or higher
- Pip (Python package installer)

## Setup Instructions

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/ai-chart-captioning.git
    cd ai-chart-captioning
    ```

2. **Install Dependencies**

    Install the required Python packages by running:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Environment Variables**

    Create a `.env` file in the root directory of the project and add your Google API key:

    ```env
    GOOGLE_API_KEY=your_google_api_key
    ```

    You can generate a Google API key for using the Gemini API from [here](https://ai.google.dev/gemini-api).

4. **Run the FastAPI Backend**

    Navigate to the `backend` directory and start the FastAPI server:

    ```bash
    cd backend
    uvicorn main:app --reload
    ```

5. **Run the Streamlit Frontend**

    Open a new terminal, navigate to the `frontend` directory, and start the Streamlit app:

    ```bash
    cd frontend
    streamlit run main.py
    ```

## Usage

1. **Upload an Image**: Use the Streamlit frontend to upload a chart image or provide a URL.
2. **Generate Caption**: The backend processes the image and generates a caption.
3. **Interact with Chatbot**: Use the chatbot interface to ask questions about the chart.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## Contact

For questions or feedback, please contact [your email address].
