from dataclasses import dataclass
from typing import Literal
import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
from dotenv import load_dotenv
import os
import pandas as pd

@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str

def load_css():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    css_file_path = os.path.join(current_dir, "..", "static", "styles.css")
    
    if not os.path.exists(css_file_path):
        st.error(f"CSS file not found at: {css_file_path}")
        return
    
    with open(css_file_path, "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "model" not in st.session_state:
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        st.session_state.model = genai.GenerativeModel('gemini-1.5-pro-latest')

def load_data():
    """Load and return the data from CSV and text files."""
    # Load and format CSV data
    csv_path = 'extracted_table.csv'
    try:
        csv_data = pd.read_csv(csv_path).to_string(index=False)
    except FileNotFoundError:
        csv_data = "CSV file not found."

    txt_path = "chart_info.txt"
    try:
        with open(txt_path, 'r') as file:
            txt_data = file.read()
    except FileNotFoundError:
        txt_data = "Text file not found."

    # Load Markdown data
    md_path = 'contextual_background_report.md'
    try:
        with open(md_path, 'r') as file:
            markdown_data = file.read()
    except FileNotFoundError:
        markdown_data = "Markdown file not found."

    return csv_data, txt_data, markdown_data

def on_click_callback():
    human_prompt = st.session_state.human_prompt

    # Load context and CSV data
    csv_data, txt_data, markdown_data = load_data()
    combined_data = f"Markdown Data:\n{markdown_data}\n\nText Data:\n{txt_data}\n\nCSV Data:\n{csv_data}"

    # Create prompt for the model with additional context
    prompt = (f"Context: {combined_data}\n"
              f"Purpose: You are an analyst helping the user understand a visual chart by providing detailed contextual information. The data provided by model is not accurate-dont say it to user, \n"
              f"User: {human_prompt}\n"
              f"Response:")

    try:
        response = st.session_state.model.generate_content(prompt)
        llm_response = response.text
    except Exception as e:
        llm_response = f"Error generating response: {str(e)}"

    st.session_state.history.append(
        Message("human", human_prompt)
    )
    st.session_state.history.append(
        Message("ai", llm_response)
    )

def chatbot_interface():
    load_css()
    initialize_session_state()

    st.title("Chatbot Interface 🤖")

    chat_placeholder = st.container()
    prompt_placeholder = st.form("chat-form")

    with chat_placeholder:
        for chat in st.session_state.history:
            div = f"""
    <div class="chat-row {'' if chat.origin == 'ai' else 'row-reverse'}">
        <div class="chat-bubble {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">{chat.message}</div>
        </div>
            """
            st.markdown(div, unsafe_allow_html=True)
        
        st.markdown("<br>" * 3, unsafe_allow_html=True)

    with prompt_placeholder:
        st.markdown("**Chat with our AI**")
        cols = st.columns((8, 2))
        cols[0].text_input(
            "Type your message...",
            value="Hello bot",
            label_visibility="collapsed",
            key="human_prompt",
        )
        cols[1].form_submit_button(
            "Send", 
            type="primary", 
            on_click=on_click_callback, 
        )

    components.html("""
    <script>
    const streamlitDoc = window.parent.document;

    const buttons = Array.from(
        streamlitDoc.querySelectorAll('.stButton > button')
    );
    const submitButton = buttons.find(
        el => el.innerText === 'Send'
    );

    streamlitDoc.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            submitButton.click();
        }
    });
    </script>
    """, 
        height=0,
        width=0,
    )
