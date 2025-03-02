from fastapi import FastAPI
from pydantic import BaseModel
from backend.query_engine import RAGChatbot
import os

app = FastAPI()

class QueryRequest(BaseModel):
    input_query: str

chatbot = RAGChatbot()

@app.post("/query/")
def query(request: QueryRequest):
    response = chatbot.generate_response(request.input_query)
    return {"response": response}
