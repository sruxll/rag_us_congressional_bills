"""RAG_LLM_Backend Model Evaluation with OpenRouter and FAISS

Evaluates a retrieval-augmented generation (RAG) system for summarizing U.S. Congressional bills using FAISS for retrieval and OpenRouter-hosted LLMs for summarization.
"""

# !pip install faiss-cpu openai sentence-transformers nltk pandas rouge-score bert_score
import os, ast, re, time, csv
import numpy as np
import pandas as pd
import requests, faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from google.colab import userdata

nltk.download("punkt")
nltk.download("stopwords")

client = OpenAI(
    api_key = userdata.get('secretName'),
    base_url="https://openrouter.ai/api/v1"
)

MODELS = [
    "deepseek/deepseek-r1:free",
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwq-32b:free",
    "google/gemini-2.0-flash-thinking-exp-1219:free",
    "deepseek/deepseek-chat-v3-0324:free"
]

bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
STOP_WORDS = set(stopwords.words("english"))

SYSTEM_PROMPT = """You are the Answering Module in a RAG system.
Your only knowledge source is the set of retrieved documents supplied with each request.
<query> the user's question </query>
<documents> a list of up to 3 text chunks, each formatted as [Document id] <content> </documents>
Instructions: Ground in documents. Cite as you go using (DocX). Admit if not enough info. Keep it concise. """

def download_file(url: str, filename: str) -> str:
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", filename)
    if not os.path.exists(path):
        r = requests.get(url); r.raise_for_status()
        with open(path, "wb") as f: f.write(r.content)
    return path

def load_files():
    idx_path = download_file(
        "https://huggingface.co/datasets/joynae5/CongressionalBillsDS/resolve/main/bill_embeddings.index",
        "bill_embeddings.index")
    csv_path = download_file(
        "https://huggingface.co/datasets/joynae5/CongressionalBillsDS/resolve/main/parsed_bills_115-119_chunks_only_embedded.csv",
        "parsed_bills.csv")
    df = pd.read_csv(csv_path)
    df["embedding"] = df["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
    return faiss.read_index(idx_path), df

faiss_index, df = load_files()

def summarize_with_openrouter(prompt, model, temperature=0.3, max_tokens=2048):
    start = time.time()
    try:
        res = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        )
        out = res.choices[0].message.content.strip()
        return out, True, time.time() - start
    except Exception as e:
        return f"ERROR: {str(e)}", False, time.time() - start

def retrieve_and_summarize_bills(query, model, top_k=3):
    q_embed = EMBED_MODEL.encode(query).reshape(1, -1)
    _, idxs = faiss_index.search(q_embed, top_k * 10)
    chunks = df.iloc[idxs[0]].copy()

    grouped = (chunks.groupby("title", group_keys=False)
        .apply(lambda g: pd.Series({
            "combined_text": "\n".join(g["text_chunk"].values[:5]),
            "measure_ids": list(g["measure_id"].unique()),
            "raw_chunks": g["text_chunk"].values[:5].tolist()}))
        .reset_index()
        .head(top_k))

    formatted_documents = "".join([f"[Doc{i+1}] {row['combined_text'][:2000]}\n\n" for i, row in grouped.iterrows()])
    prompt = f"""<query>{query}</query>\n\n<documents>\n{formatted_documents}</documents>"""
    summary, success, time_taken = summarize_with_openrouter(prompt, model)
    return summary, grouped, success, time_taken

def hallucination_results(results_df):
    cos_scores, bert_scores = [], []
    for _, row in results_df.iterrows():
        if not row["success"]:
            cos_scores.append(None); bert_scores.append(None); continue
        try:
            summary = row["response_text"]
            source_chunks = row.get("raw_chunks", [])
            if not summary or not source_chunks:
                cos_scores.append(None); bert_scores.append(None); continue
            summary_emb = EMBED_MODEL.encode(summary)
            source_embs = EMBED_MODEL.encode(source_chunks)
            cos_scores.append(cosine_similarity([summary_emb], source_embs).mean())
            _, _, F1 = bert_scorer.score([summary], [" ".join(source_chunks)])
            bert_scores.append(float(F1[0]))
        except:
            cos_scores.append(None); bert_scores.append(None)
    results_df["cosine_similarity"] = cos_scores
    results_df["bert_score"] = bert_scores
    return results_df

def run_evaluations():
    questions = [
        {"id": 1, "category": "Environment", "query": "What bills address climate change and renewable energy?"},
        {"id": 2, "category": "Healthcare", "query": "What legislation exists for healthcare access and affordability?"},
        {"id": 3, "category": "Immigration", "query": "What policies exist regarding border security and immigration reform?"},
        {"id": 4, "category": "Technology", "query": "How is Congress addressing AI regulation and data privacy?"},
        {"id": 5, "category": "Education", "query": "What bills exist for student loan forgiveness and education funding?"}
    ]

    results = []
    with open("model_evaluation_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question_id", "category", "query", "model",
            "success", "response_time", "bert_score",
            "cosine_similarity", "summary_length", "response_text"])
        writer.writeheader()

    for q in questions:
        print(f"\nEvaluating: {q['query']}")
        for model in MODELS:
            print(f"Model: {model}")
            summary, grouped, success, elapsed = retrieve_and_summarize_bills(q["query"], model)
            chunks = [c for row in grouped["raw_chunks"] for c in row] if success else []
            temp_df = pd.DataFrame([{"response_text": summary, "raw_chunks": chunks, "success": success}])
            scored = hallucination_results(temp_df).iloc[0]

            result = {
                "question_id": q["id"], "category": q["category"], "query": q["query"], "model": model,
                "success": success, "response_time": round(elapsed, 2),
                "bert_score": round(scored["bert_score"], 3) if scored["bert_score"] else 0,
                "cosine_similarity": round(scored["cosine_similarity"], 3) if scored["cosine_similarity"] else 0,
                "summary_length": len(summary) if isinstance(summary, str) else 0,
                "response_text": summary[:500] if isinstance(summary, str) else "ERROR"
            }

            results.append(result)
            with open("model_evaluation_results.csv", "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=result.keys())
                writer.writerow(result)

    return results

def display_summary(results):
    if not results: return print("No results.")
    df = pd.DataFrame(results)
    print("\n EVALUATION SUMMARY")
    print("\nSuccess Rate\n", df.groupby("model")["success"].mean() * 100)
    print("\nBERTScore\n", df[df["success"]].groupby("model")["bert_score"].mean())
    print("\nCosine Similarity \n", df[df["success"]].groupby("model")["cosine_similarity"].mean())
    print("\nResponse Time \n", df.groupby("model")["response_time"].mean())
    print("\nSaved to model_evaluation_results.csv")

if __name__ == "__main__":
    results = run_evaluations()
    display_summary(results)
