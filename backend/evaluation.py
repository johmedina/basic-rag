import pandas as pd
import time
import openai
from backend.retrieval import DocumentRetriever
from backend.query_engine import RAGChatbot
from rouge_score import rouge_scorer
from fuzzywuzzy import fuzz

def load_ground_truth(csv_path="data/ground_truth_gpt.csv"):
    """Loads the ground truth Q&A pairs from the provided CSV file."""
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")

def measure_latency(chatbot, ground_truth):
    """Measures chatbot response latency."""
    latencies = []
    queries = [qa["Question"] for qa in ground_truth]
    
    for i, query in enumerate(queries):
        start_time = time.time()
        _ = chatbot.generate_response(query)
        end_time = time.time()
        
        latencies.append(end_time - start_time)
    
    avg_latency = sum(latencies) / len(latencies)
    print(f"Average Response Time: {avg_latency:.2f} seconds")
    return avg_latency

def evaluate_faithfulness(chatbot, retriever, ground_truth):
    """Uses GPT-4 to evaluate factual consistency between retrieved docs and generated answers."""
    client = openai.OpenAI()
    scores = []
    
    for qa in ground_truth:
        query, expected_answer = qa["Question"], qa["Answer"]
        retrieved_docs = retriever.hybrid_retrieval(query, top_k=3)
        response = chatbot.generate_response(query)
        
        prompt = f"""
            Given the retrieved documents: {retrieved_docs}, evaluate the factual consistency of the answer: "{response}".
            Provide a score between 1 (low) and 5 (high) based on whether the answer is factually accurate and grounded 
            in the retrieved documents.
        """
        
        gpt_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        
        try:
            print('FAITHFULNESS REPSONSE ---', gpt_response, '---')
            score = float(gpt_response.choices[0].message.content.strip())
        except ValueError:
            score = 1.0  
        scores.append(score)
    
    avg_faithfulness = sum(scores) / len(scores)
    print(f"Average Faithfulness Score: {avg_faithfulness:.2f}")
    return avg_faithfulness

def rouge_l_score(expected, response):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(expected, response)["rougeL"].fmeasure

def fuzzy_match(expected, response):
    return fuzz.token_sort_ratio(expected, response) / 100  

def evaluate_retrieval(retriever, ground_truth, top_k=5):
    """Evaluates retrieval accuracy using Recall@K."""
    correct_retrievals = 0
    total_questions = len(ground_truth)
    
    for qa in ground_truth:
        query, expected_answer = qa["Question"], qa["Answer"]
        retrieved_docs = retriever.hybrid_retrieval(query, top_k)
        
        if any(expected_answer in doc for doc in retrieved_docs):
            correct_retrievals += 1
    
    recall_at_k = correct_retrievals / total_questions
    print(f"Recall@{top_k}: {recall_at_k:.2f}")
    return recall_at_k

def evaluate_generation(chatbot, ground_truth):
    """Evaluates chatbot responses using ROUGE-L and Fuzzy Matching."""
    rouge_scores, fuzzy_scores = [], []
    
    for i, qa in enumerate(ground_truth):
        query, expected_answer = qa["Question"], qa["Answer"]
        response = chatbot.generate_response(query)
        response = response.split("My final answer is")[1]
        print('EXPECTED ANSWER ---', expected_answer, '---', 'RESPONSE ---', response, '---')
        
        rouge_scores.append(rouge_l_score(expected_answer, response))
        fuzzy_scores.append(fuzzy_match(expected_answer, response))
    
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    avg_fuzzy = sum(fuzzy_scores) / len(fuzzy_scores)
    
    print(f"ROUGE-L Score: {avg_rouge:.2f}, Fuzzy Match: {avg_fuzzy:.2f}")
    return avg_rouge, avg_fuzzy




if __name__ == "__main__":
    ground_truth = load_ground_truth("data/ground_truth_gpt.csv")
    retriever = DocumentRetriever()
    chatbot = RAGChatbot()
    
    # print("Evaluating Retrieval...")
    # evaluate_retrieval(retriever, ground_truth, top_k=5)
    
    print("Evaluating Response Generation...")
    evaluate_generation(chatbot, ground_truth)
    
    # print("Measuring Latency...")
    # measure_latency(chatbot, ground_truth)
    
    # print("Evaluating Faithfulness...")
    # evaluate_faithfulness(chatbot, retriever, ground_truth)
