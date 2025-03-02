import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from openai import OpenAI

class AgenticRetriever:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2", index_path="data/embeddings/index.faiss"):
        self.client = OpenAI()
        self.model = SentenceTransformer(embedding_model)
        self.index_path = index_path
        self.index = None
        self.text_data = []
        self.load_index()

    def agentic_query_expansion(self, query):
        """ Uses an LLM agent to improve query formulation """
        system_prompt = "You are an expert in financial data retrieval. Rewrite the query for better document retrieval."
        gpt_response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=50
        )
        return gpt_response.choices[0].message.content.strip()

    def create_index(self, documents):
        """Encodes documents and creates FAISS index."""
        embeddings = self.model.encode(documents, convert_to_numpy=True)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_path)

        self.text_data = documents
        with open("data/embeddings/documents.json", "w") as f:
            json.dump(self.text_data, f)

    def load_index(self):
        """Loads FAISS index if it exists, else creates a new one."""
        if os.path.exists(self.index_path):
            print("Loading FAISS index...")
            self.index = faiss.read_index(self.index_path)
            with open("data/embeddings/documents.json", "r") as f:
                self.text_data = json.load(f)
        else:
            print("FAISS index not found. Creating a new one...")
            with open("data/embeddings/text_data.json", "r") as f:
                text_data = json.load(f)
            
            if not text_data:
                text_data = ["No valid documents found. Please check preprocessing."]
            
            self.create_index(text_data)

    def retrieve(self, query, top_k=5):
        """Expands query using an agent and retrieves relevant text chunks."""
        improved_query = self.agentic_query_expansion(query)
        print(f"Expanded Query: {improved_query}")

        query_embedding = self.model.encode([improved_query], convert_to_numpy=True)
        document_embeddings = np.load("data/embeddings/text_embeddings.npy")

        similarities = np.dot(document_embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        retrieved_chunks = [self.text_data[i] for i in top_indices if i < len(self.text_data)]
        return retrieved_chunks


if __name__ == "__main__":
    retriever = AgenticRetriever()
    retriever.load_index()
    print(retriever.retrieve("financial report analysis"))
