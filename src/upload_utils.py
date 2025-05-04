import pymupdf
from openai import OpenAI
import streamlit as st

# Extracts all text from a PDF file
def extract_text_from_pdf(pdf_file):
    doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

# Extracts all text from a plain text file
def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

# Splits a long text into overlapping chunks for better AI processing
def chunk_text(text, chunk_size=300, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Uses OpenAI to generate a short, readable title for a given text chunk
def generate_chunk_title(text):
    client = OpenAI(api_key=st.session_state.get("openai_api_key"))
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Summarize this content into a short, descriptive title (5-10 words max)."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=20
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Chunk title generation failed: {e}")
        return text[:80] + "..."