import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Backend URLs
UPLOAD_URL = "http://127.0.0.1:8000/upload_image/"
UPLOAD_FROM_URL_URL = "http://127.0.0.1:8000/upload_image_from_url/"
VALIDATE_URL = "http://127.0.0.1:8000/validate_url/"

def main():
    st.title("Chart Captioning Application")

    # File uploader for image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Upload the file to the FastAPI backend
        if st.button("Upload"):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(UPLOAD_URL, files=files)

            if response.status_code == 200:
                st.success("File uploaded successfully!")
            else:
                st.error(f"File upload failed: {response.json()}")

    # URL input for validation and download
    url = st.text_input("Enter image URL")

    if url:
        # Preview the image from the URL
        try:
            image_response = requests.get(url)
            if image_response.status_code == 200:
                image = Image.open(BytesIO(image_response.content))
                st.image(image, caption="Image Preview", use_column_width=True)
                st.write("URL preview successful!")
            else:
                st.error(f"Failed to retrieve image from URL: {image_response.status_code}")
        except Exception as e:
            st.error(f"Error retrieving image: {e}")

        if st.button("Save Image from URL"):
            response = requests.post(UPLOAD_FROM_URL_URL, json={"url": url})
            if response.status_code == 200:
                st.success("Image downloaded and saved successfully!")
            else:
                st.error(f"Image download failed: {response.json()}")

if __name__ == "__main__":
    main()
