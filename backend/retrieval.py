import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class DocumentRetriever:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2", index_path="data/embeddings/index.faiss"):
        self.model = SentenceTransformer(embedding_model)
        self.index_path = index_path
        self.index = None
        self.text_data = []
        self.load_index()

    def create_index(self, documents):
        embeddings = self.model.encode(documents, convert_to_numpy=True)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_path)

        self.text_data = documents
        with open("data/embeddings/documents.json", "w") as f:
            json.dump(self.text_data, f)

    def load_index(self):
        """Load FAISS index if it exists, else create a new one."""
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

    def retrieve(self, query, top_k=10):
        """
        Retrieves the top_k most relevant text chunks based on semantic similarity.
        """
        with open("data/embeddings/text_data.json", "r") as f:
            text_data = json.load(f)

        document_embeddings = np.load("data/embeddings/text_embeddings.npy")
        query_embedding = self.model.encode(
            [query], 
            convert_to_numpy=True, 
            normalize_embeddings=True 
        )

        similarities = cosine_similarity(query_embedding, document_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        retrieved_chunks = [text_data[i] for i in top_indices if i < len(text_data)]
        return retrieved_chunks


if __name__ == "__main__":
    retriever = DocumentRetriever()
    retriever.load_index()
    print(retriever.hybrid_retrieval("financial report analysis"))
