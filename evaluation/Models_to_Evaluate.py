"""RAG_LLM_Backend Model Evaluation with OpenRouter and FAISS

Evaluates a retrieval-augmented generation (RAG) system for summarizing U.S. Congressional bills using FAISS for retrieval and OpenRouter-hosted LLMs for summarization.
"""

# Install required libraries
!pip install faiss-cpu
!pip install openai

import os
import pandas as pd
import numpy as np
import faiss
import ast
import re
from sentence_transformers import SentenceTransformer
import requests
from tqdm import tqdm
from openai import OpenAI

# Set up OpenRouter API key and client
client = OpenAI(
    api_key=<"your-key-here">,
    base_url="https://openrouter.ai/api/v1"
)

# Choose a free OpenRouter model for summarization
"""
Free models can be found here: https://openrouter.ai/models/?q=free&max_price=0

***Models to evaluate with over a billion tokens***
- deepseek/deepseek-r1:free
- google/gemini-2.0-flash-exp:free
- qwen/qwq-32b:free
- google/gemini-2.0-flash-thinking-exp-1219:free
- deepseek/deepseek-chat-v3-0324:free
"""
llm_use = "qwen/qwq-32b:free"

def download_file(url: str, filename: str) -> str:
    """
    Downloads a file from a URL to a local data directory if not already present.

    Args:
        url (str): URL to the file.
        filename (str): Filename to save locally under the 'data/' folder.

    Returns:
        str: Local path to the downloaded file.
    """
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", filename)
    if not os.path.exists(path):
        r = requests.get(url)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path

def load_files():
    """
    Downloads and loads the FAISS index and preprocessed Congressional bill embeddings.

    Returns:
        faiss.Index: The loaded FAISS index object.
        pd.DataFrame: DataFrame containing bill metadata and text chunks.
    """
    index_url = "https://huggingface.co/datasets/joynae5/CongressionalBillsDS/resolve/main/bill_embeddings.index"
    csv_url = "https://huggingface.co/datasets/joynae5/CongressionalBillsDS/resolve/main/parsed_bills_115-119_chunks_only_embedded.csv"

    idx_path = download_file(index_url, "bill_embeddings.index")
    csv_path = download_file(csv_url, "parsed_bills.csv")

    faiss_index = faiss.read_index(idx_path)
    df = pd.read_csv(csv_path)

    # Convert string representation of embeddings back to arrays
    df["embedding"] = df["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))

    return faiss_index, df

# Load index and DataFrame
faiss_index, df = load_files()

def summarize_with_openrouter(prompt, model=llm_use):
    """
    Uses OpenRouter to summarize bill text using the specified language model.

    Args:
        prompt (str): Prompt to send to the LLM.
        model (str): OpenRouter model identifier.

    Returns:
        str: Textual summary from the model.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful legislative analyst. Summarize each bill clearly. Then, list the corresponding citations (measure_ids) for each bill mentioned."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.5,
        max_tokens=2048
    )
    return response.choices[0].message.content

def retrieve_and_summarize_bills(query, top_k=3):
    """
    Retrieves the most relevant bills from the FAISS index based on query,
    summarizes the top `top_k` using a large language model (LLM).

    Args:
        query (str): User input query describing their information need.
        top_k (int): Number of top relevant bill clusters to summarize.

    Returns:
        str: Structured multi-bill summary from the LLM.
    """
    # Load embedding model to encode the query
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embed_model.encode(query).reshape(1, -1)

    # Perform similarity search
    _, indices = faiss_index.search(query_embedding, top_k * 10)
    retrieved_chunks = df.iloc[indices[0]].copy()

    # Group text chunks by title + measure_id for coherent summary input
    retrieved_chunks["group_key"] = retrieved_chunks["title"] + " | " + retrieved_chunks["measure_id"]

    grouped = (
        retrieved_chunks.groupby("title", group_keys=False)
        .apply(lambda g: pd.Series({
            "combined_text": "\n".join(g["text_chunk"].values[:5]),  # Combine top 5 chunks
            "measure_ids": list(g["measure_id"].unique())
        }),
        include_groups=False
        )
        .reset_index()
        .head(top_k)
    )

    # Build a clean, structured prompt for the LLM
    prompt = "You are a helpful legislative analyst. For each numbered bill, provide:\n1. The bill title\n2. A 1–2 sentence summary\n3. A list of measure_ids\n\n"

    for idx, row in grouped.iterrows():
        title = row["title"]
        measure_ids = ", ".join(row["measure_ids"])
        snippet = row["combined_text"][:2000]  # Clip for token limit
        prompt += f"{idx+1}. {title}\nText:\n{snippet}\nMeasure IDs: {measure_ids}\n\n"

    # Call LLM once with all grouped bills
    summary_output = summarize_with_openrouter(prompt)

    return summary_output

# Run test query when script is executed directly
if __name__ == "__main__":
    query = "What is being done to handle invasive animals?"
    print(retrieve_and_summarize_bills(query, top_k=3))
