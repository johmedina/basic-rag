import fitz
import pandas as pd
import os
import json
from PIL import Image
from io import BytesIO
import numpy as np
import pytesseract
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],  
    )
    return text_splitter.split_text(text)

def extract_text(pdf_path):
    """
    Extracts text from a PDF while preserving page structure.
    """
    doc = fitz.open(pdf_path)
    chunks = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            page_chunks = chunk_text(text, chunk_size=500, overlap=50)
            for chunk in page_chunks:
                chunks.append({"text": chunk, "page": page_num})
    
    return chunks

def extract_images_text(pdf_path):
    """
    Extracts text from images in a PDF using OCR.
    """
    doc = fitz.open(pdf_path)
    image_texts = []
    
    for page_num, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes))
            extracted_text = pytesseract.image_to_string(image)
            
            if extracted_text.strip():
                image_texts.append({"text": extracted_text, "page": page_num, "type": "image"})
    
    return image_texts

def extract_tables(pdf_path):
    """
    Extracts tables from a PDF using pdfplumber (better accuracy than BeautifulSoup).
    """
    tables = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            extracted_tables = page.extract_tables()
            
            for table in extracted_tables:
                df = pd.DataFrame(table)
                table_text = df.to_string(index=False, header=False)
                tables.append({"text": table_text, "page": page_num, "type": "table"})
    
    return tables

def preprocess_pdfs(pdf_folder):
    """
    Preprocesses PDFs by extracting text, images, and tables into structured format.
    """
    all_data = []
    
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path}...")
        
        processed_data = (
            extract_text(pdf_path) + 
            extract_images_text(pdf_path) + 
            extract_tables(pdf_path)  
        )
        
        all_data.extend(processed_data)
    
    # Save structured JSON
    with open("data/processed_data.json", "w") as f:
        json.dump(all_data, f, indent=4)
    
    return all_data

def save_embeddings(data, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generates embeddings and saves them in FAISS.
    """
    os.makedirs("data/embeddings", exist_ok=True)
    model = SentenceTransformer(model_name)
    
    all_texts = [entry["text"] for entry in data]
    embeddings = model.encode(all_texts, convert_to_numpy=True)

    np.save("data/embeddings/text_embeddings.npy", embeddings)

    with open("data/embeddings/text_data.json", "w") as f:
        json.dump(data, f)

    print(f"Embeddings saved for {len(all_texts)} chunks.")

if __name__ == "__main__":
    pdf_folder = "data/documents/"
    structured_data = preprocess_pdfs(pdf_folder)
    save_embeddings(structured_data)
