from backend.retrieval import DocumentRetriever
from openai import OpenAI
from backend.config import OPENAI_API_KEY
import os

class RAGChatbot:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.retriever = DocumentRetriever()
        self.retriever.load_index() 

    def generate_response(self, query):
        retrieved_docs = self.retriever.hybrid_retrieval(query)
        if not retrieved_docs:
            return "Sorry, I couldn't find relevant information."
        
        prompt = f"Given the following retrieved information: {retrieved_docs}, answer the query: {query}."
        # print('PROMPT --- ', prompt, '---')
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": 
                  """You are an AI assistant for the Ooredoo Group. You will be answering mostly financial related questions.
                  Your answer should end with: My final answer is __."""
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()

