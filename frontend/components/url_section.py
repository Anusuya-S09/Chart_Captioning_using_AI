import streamlit as st
import requests,httpx

def enter_url():
    st.sidebar.markdown('<div class="sidebar"> <div class="section">', unsafe_allow_html=True)
    url = st.sidebar.text_input("Enter image URL")
    if st.sidebar.button("Validate URL"):
        try:
            response = requests.post("http://127.0.0.1:8000/validate_url/", json={"url": url})
            response.raise_for_status()
            st.success("URL is valid and points to an image!")

            st.image(url, caption="Image from URL", use_column_width=True)

            process_response = requests.post("http://127.0.0.1:8000/process_image/", json={"url": url})
            process_response.raise_for_status()
            processed_image_url = process_response.json().get("processed_image_url")
            st.image(processed_image_url, caption="Processed Image", use_column_width=True)

            with st.spinner('Processing...'):
                with httpx.Client(timeout=80) as client:           
                    response = requests.post(
                        "http://127.0.0.1:8000/validate_url/",
                        json={"url": url},
                        headers={"Content-Type": "application/json"}
                    )


            generate_response = requests.post("http://127.0.0.1:8000/generate_caption/", json={"url": processed_image_url})
            generate_response.raise_for_status()

            return True  # Indicate successful data extraction

        except requests.RequestException as e:
            st.markdown(f'<p class="error">An error occurred: {e}</p>', unsafe_allow_html=True)
    st.markdown('</div> </div>', unsafe_allow_html=True)
    return False  # Indicate failure
