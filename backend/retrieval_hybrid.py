import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from backend.graph_rag import KnowledgeGraph
from backend.config import OPENAI_API_KEY
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi 

class HybridAgenticRetriever:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2", index_path="data/embeddings/index.faiss"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = SentenceTransformer(embedding_model)
        self.index_path = index_path
        self.index = None
        self.text_data = []
        self.knowledge_graph = KnowledgeGraph()
        self.bm25 = None
        self.load_index()

    def agentic_query_expansion(self, query):
        system_prompt = """You are an AI specializing in financial data retrieval. 
            Improve the search query to retrieve the most relevant documents while preserving financial terminology.
            Do NOT change financial terms like H1, Q1, Q2, YoY, EBITDA, Capex, Net Profit, Revenue, or similar domain-specific language.
            """
        gpt_response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=200
        )
        return gpt_response.choices[0].message.content.strip()

    def create_index(self, documents):
        """Encodes documents and creates FAISS and BM25 indices."""
        text_chunks = [doc["text"] for doc in documents if "text" in doc] 

        embeddings = self.model.encode(text_chunks, convert_to_numpy=True)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_path)

        self.text_data = documents
        self.bm25 = BM25Okapi([chunk.split() for chunk in text_chunks])

        with open("data/embeddings/documents.json", "w") as f:
            json.dump(self.text_data, f)

        print(f"FAISS and BM25 indexes created with {len(text_chunks)} documents.")


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

        self.bm25 = BM25Okapi([doc["text"].split() for doc in self.text_data if "text" in doc])

    def retrieve_faiss(self, query, top_k=5):
        """Performs FAISS-based retrieval."""
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        document_embeddings = np.load("data/embeddings/text_embeddings.npy")

        similarities = cosine_similarity(query_embedding, document_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        retrieved_chunks = [self.text_data[i] for i in top_indices if i < len(self.text_data)]
        return retrieved_chunks

    def retrieve_bm25(self, query, top_k=5):
        """Performs BM25 keyword-based retrieval."""
        if not self.bm25:
            return [] 

        query_tokens = query.split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]

        retrieved_chunks = [self.text_data[i] for i in top_indices if i < len(self.text_data)]
        return retrieved_chunks

    def retrieve_graph(self, query):
        """Retrieves related financial entities from the knowledge graph and ranks them."""
        entity = query.split()[0]  # Extract first word as entity (basic approach)
        related_entities = self.knowledge_graph.query_graph(entity)

        # Rank related entities by relevance
        ranked_entities = self.rank_entities(query, related_entities)
        
        return ranked_entities[:3]  # Return top 3 most relevant entities

    def rank_entities(self, query, entities):
        """Ranks entities based on their relevance to the query."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        entity_embeddings = self.model.encode(entities, convert_to_numpy=True)

        similarities = np.dot(entity_embeddings, query_embedding.T).flatten()
        ranked_indices = np.argsort(similarities)[::-1]
        
        return [entities[i] for i in ranked_indices]

    def hybrid_retrieve(self, query, top_k=15):
        """Combines FAISS, BM25, and Graph RAG retrieval."""
        faiss_results = self.retrieve_faiss(query, top_k)
        bm25_results = self.retrieve_bm25(query, top_k)

        combined_results = {json.dumps(doc, sort_keys=True) for doc in faiss_results + bm25_results}
        combined_results = [json.loads(doc) for doc in combined_results] 

        # Use Graph RAG only if FAISS + BM25 doesn’t return enough results
        if len(combined_results) < top_k:
            improved_query = self.agentic_query_expansion(query)
            print("ORIGINAL QUERY ---", query, '---', 'LLM QUERY ---', improved_query, '---')
            graph_results = self.retrieve_graph(improved_query)
            combined_results += graph_results[:2]  

        return combined_results 


if __name__ == "__main__":
    retriever = HybridAgenticRetriever()
    retriever.load_index()
    print(retriever.hybrid_retrieve("Ooredoo revenue"))
