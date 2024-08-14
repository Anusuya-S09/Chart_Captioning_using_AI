import streamlit as st
import httpx
import requests

def upload_image():
    st.markdown('<div class="sidebar"> <div class="section">', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        st.sidebar.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        if st.sidebar.button("Upload Image"):
            try:
                # Upload the image
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post("http://127.0.0.1:8000/upload_image/", files=files)
                response.raise_for_status()
                st.success("File uploaded successfully!")

                # Process the image
                process_response = requests.get(f"http://127.0.0.1:8000/process_image/{uploaded_file.name}")
                process_response.raise_for_status()
                processed_image_path = process_response.json().get("processed_image_path")

                # Reset the file pointer to the beginning
                uploaded_file.seek(0)

                # Send the image to the extract endpoint
                with st.spinner('Processing...'):
                    with httpx.Client(timeout=1800) as client:
                        response = client.post(
                            "http://127.0.0.1:8000/extract",
                            files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                        )

                # Generate the caption
                with open(processed_image_path, "rb") as f:
                    generate_response = requests.post(
                        "http://127.0.0.1:8000/generate_caption/",
                        files={"file": (uploaded_file.name, f, 'image/png')}
                    )
                    generate_response.raise_for_status()

                return True  # Indicate successful data extraction

            except requests.RequestException as e:
                st.markdown(f'<p class="error">An error occurred: {e}</p>', unsafe_allow_html=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
                return False  # Indicate failure
