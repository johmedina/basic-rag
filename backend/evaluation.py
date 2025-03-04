import pandas as pd
import time
import openai
from backend.retrieval import DocumentRetriever
from backend.retrieval_agent import AgenticRetriever
from backend.retrieval_hybrid import HybridAgenticRetriever
from backend.query_engine import RAGChatbot
from rouge_score import rouge_scorer
from fuzzywuzzy import fuzz

def load_ground_truth(csv_path="data/ground_truth_gpt.csv"):
    """Loads the ground truth Q&A pairs from the provided CSV file."""
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")

def load_evaluation_data(csv_path="data/results/chatbot_analysis_hybrid_5.csv"):
    """Loads saved chatbot responses and ground truth data from the CSV file."""
    df = pd.read_csv(csv_path)
    return df

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
    """Uses GPT-4o-mini to evaluate factual consistency between retrieved docs and generated answers."""
    client = openai.OpenAI()
    scores = []
    
    for qa in ground_truth:
        query, expected_answer = qa["Question"], qa["Answer"]
        retrieved_docs = retriever.hybrid_retrieve(query, top_k=5)
        response = chatbot.generate_response(query)
        
        prompt = f"""
            Given the retrieved documents: {retrieved_docs}, and expected answer {expected_answer},
            evaluate the factual consistency of the generated answer: "{response}".
            Provide a score between 1 (low) and 5 (high) based on whether the answer is factually accurate and grounded 
            in the retrieved documents. Just answer with a number.
        """
        
        gpt_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        
        try:
            score = float(gpt_response.choices[0].message.content.strip())
            print('FAITHFULNESS REPSONSE ---', score, '---')
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


def evaluate_retrieval(retriever, ground_truth, top_k=10):
    """Evaluates retrieval accuracy using Recall@K."""
    correct_retrievals = 0
    total_questions = len(ground_truth)
    
    for i, qa in enumerate(ground_truth):
        query, expected_answer = qa["Question"], qa["Answer"]
        retrieved_docs = retriever.hybrid_retrieve(query, top_k)
        # print('RETRIEVED DOCS ---', retrieved_docs, '---')
        # print('EXPECTED ---', expected_answer, '---')
        
        if any(expected_answer in doc for doc in retrieved_docs):
            correct_retrievals += 1
    
    recall_at_k = correct_retrievals / total_questions
    print(f"Recall@{top_k}: {recall_at_k:.2f}")
    return recall_at_k


def evaluate_generation(df):
    """Evaluates chatbot responses using ROUGE-L and Fuzzy Matching."""
    def preprocess_response(response):
        """Extracts the chatbot's final response if it contains 'My final answer is'."""
        if "My final answer is" in response:
            return response.split("My final answer is")[1].strip()
        return response.strip()

    def rouge_apply(row):
        processed_response = preprocess_response(row["Chatbot Answer"])
        return rouge_l_score(row["Ground Truth Answer"], processed_response)
    
    def fuzzy_apply(row):
        processed_response = preprocess_response(row["Chatbot Answer"])
        return fuzzy_match(row["Ground Truth Answer"], processed_response)
      
    rouge_scores = df.apply(rouge_apply, axis=1)
    fuzzy_scores = df.apply(fuzzy_apply, axis=1)

    avg_rouge = rouge_scores.mean()
    avg_fuzzy = fuzzy_scores.mean()
    
    print(f"ROUGE-L Score: {avg_rouge:.2f}, Fuzzy Match: {avg_fuzzy:.2f}")
    return avg_rouge, avg_fuzzy


def human_analysis(chatbot, ground_truth, filename="data/results/chatbot_analysis_hybrid_5.csv"):
    """Record ground truth answers and chatbot answers for human analysis"""
    data = []
    
    for i, qa in enumerate(ground_truth):
        query, expected_answer = qa["Question"], qa["Answer"]
        response = chatbot.generate_response(query)
        data.append([query, expected_answer, response])
    
    df = pd.DataFrame(data, columns=["Question", "Ground Truth Answer", "Chatbot Answer"])
    df.to_csv(filename, index=False, encoding="utf-8")
    
    print(f"Analysis saved to {filename}")

if __name__ == "__main__":
    ground_truth = load_ground_truth("data/ground_truth_gpt.csv")
    retriever = HybridAgenticRetriever()
    chatbot = RAGChatbot()
    
    print("Evaluating Retrieval...")
    evaluate_retrieval(retriever, ground_truth)
    
    print("Evaluating Response Generation...")
    df = load_evaluation_data()
    evaluate_generation(df)
    
    # print("Measuring Latency...")
    # measure_latency(chatbot, ground_truth)

    # print("Human analysis csv creation")
    # human_analysis(chatbot, ground_truth)
    
    # print("Evaluating Faithfulness...")
    # evaluate_faithfulness(chatbot, retriever, ground_truth)
