import streamlit as st
from components.upload_section import upload_image
from components.url_section import enter_url
from components.chatbot_component import chatbot_interface
import os
import shutil

# Define the style to be used across the app
def load_css():
    st.markdown("""
        <style>
        body {
            font-family: 'Arial', sans-serif;
            background-image: ""
        }
        .sidebar .sidebar-content {
            padding: 20px;
        }
        .main-content {
            padding: 20px;
        }
        .section {
            margin-bottom: 10px;
            padding: 20px;
            border-radius: 10px;
        }
        .section h2 {
            margin-bottom: 10px;
            font-size: 24px;
        }
        .result-image {
            margin-top: 20px;
            display: block;
            margin-left: auto;
            margin-right: auto;
            max-width: 100%;
        }
        .markdown-text {
            border: 1px #ddd;
            border-radius: 5px;
            padding: 10px;
            background-color: #f9f9f9;
        }
        .error {
            color: #ff4d4d;
            font-weight: bold;
        }
        .success {
            color: #4dff4d;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    load_css()

    # Initialize session state variables
    if 'data_extracted' not in st.session_state:
        st.session_state.data_extracted = False

    st.sidebar.title("GraphixAI")
    st.sidebar.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<h2>Upload Your Chart</h2>', unsafe_allow_html=True)
    if upload_image():
        st.session_state.data_extracted = True

    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.header("Caption")

    # Display the caption and chat only if data has been extracted
    if st.session_state.data_extracted:
        try:
            with open("chart_info.txt", "r") as file:
                caption = file.read()
                if caption:
                    st.markdown('<div class="section">', unsafe_allow_html=True)
                    st.markdown(caption, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="error">No caption available. Please upload an image or enter a URL.</p>', unsafe_allow_html=True)
        except FileNotFoundError:
            st.markdown('<p class="error">No caption file found. Data is not clear.</p>', unsafe_allow_html=True)

        # Chat feature
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        chatbot_interface()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="error">Please upload an image to generate a caption and chat.</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)



if __name__ == "__main__":
    main()
