import fitz 
import pandas as pd
import os
import json
from PIL import Image
from io import BytesIO
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytesseract
from bs4 import BeautifulSoup

def chunk_text(text, chunk_size=500, overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],  
    )
    return text_splitter.split_text(text)

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = " ".join([page.get_text("text") for page in doc])
    return chunk_text(full_text, chunk_size=500, overlap=50)  


def extract_images(pdf_path, output_dir="data/images"):
    doc = fitz.open(pdf_path)
    os.makedirs(output_dir, exist_ok=True)
    image_texts = []
    for i, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes))
            image_path = os.path.join(output_dir, f"{os.path.basename(pdf_path).replace('.pdf', '')}_page_{i}_img_{img_index}.png")
            image.save(image_path)
            extracted_text = pytesseract.image_to_string(image)
            if extracted_text.strip():
                image_texts.append(f"Extracted from image {image_path}: {extracted_text}")
    return image_texts

def extract_tables(pdf_path):
    doc = fitz.open(pdf_path)
    tables = []
    for page in doc:
        html_text = page.get_text("html")
        soup = BeautifulSoup(html_text, "html.parser")
        table_elements = soup.find_all("table")
        for table in table_elements:
            try:
                df = pd.read_html(str(table))[0]
                table_data = df.to_dict()
                formatted_table = json.dumps(table_data, indent=4)
                tables.append(formatted_table)
            except Exception:
                continue
    return tables

def preprocess_pdfs(pdf_folder):
    all_texts = []
    all_images = []
    all_tables = []
    
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path}...")
        processed_data = {
            "text": extract_text(pdf_path),
            "images": extract_images(pdf_path),
            "tables": extract_tables(pdf_path)
        }
        all_texts.append(processed_data["text"])
        all_images.extend(processed_data["images"])
        all_tables.extend(processed_data["tables"])
    
    combined_data = {"texts": all_texts, "images": all_images, "tables": all_tables}
    with open("data/processed_data.json", "w") as f:
        json.dump(combined_data, f, indent=4)

    processed_data = all_texts + all_images + all_tables
    
    return processed_data

def save_embeddings(text_data, model_name="sentence-transformers/all-MiniLM-L6-v2", index_path="data/embeddings/index.faiss"):
    os.makedirs("data/embeddings", exist_ok=True)
    model = SentenceTransformer(model_name)

    all_chunks = []
    for doc in text_data:
        all_chunks.extend(doc)  # Store each chunk separately

    embeddings = model.encode(all_chunks, convert_to_numpy=True)
    np.save("data/embeddings/text_embeddings.npy", embeddings)

    with open("data/embeddings/text_data.json", "w") as f:
        json.dump(all_chunks, f)

    print(f"Embeddings saved for {len(all_chunks)} chunks.")


if __name__ == "__main__":
    pdf_folder = "data/documents/"
    texts = preprocess_pdfs(pdf_folder)
    save_embeddings(texts)
