import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import ast  # To safely evaluate string representations of lists
from tqdm import tqdm
from nltk.corpus import wordnet
from transformers import AutoTokenizer
import torch
"""Make sure to upload "parsed_bills_115-119_chunks_only_embedded.csv and "bill_embeddings.index in working directory"""


# Function to clear GPU memory
def clear_gpu_memory():
    torch.cuda.empty_cache()


# Load Sentence Transformer for embeddings will use GPU. switch to "cpu" if needed
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda") 

# Initialize tokenizer for summarizer model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn") 

# Load Summarization Model
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn", # Adjust model if needed
    device="cuda",            # will use GPU. switch to "cpu" if needed
    torch_dtype="auto",      # helps auto-switch to float16
    max_length=150,          # Control max output length
    min_length=30            # Control minimum output length
)

# Load LLM 
llm = pipeline("text-generation",
               model="gpt2", # Adjust model if needed
               device="cuda",              # will use GPU. switch to "cpu" if needed
               torch_dtype="auto",        # helps auto-switch to float16
               max_new_tokens=150)        # keep this in check

# Load FAISS Index
faiss_index = faiss.read_index("bill_embeddings.index")

# Load Data
file_path = "parsed_bills_115-119_chunks_only_embedded.csv"
df = pd.read_csv(file_path)

# Convert embeddings from CSV to NumPy Arrays
df["embedding"] = df["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))


def retrieve_and_summarize_bills(query, top_k=5):
    """Retrieve the most relevant bills, group their chunks, and summarize them.

    Args:
        query (str): User query.
        top_k (int, optional): Retrieve top most relevant bills. Defaults to 5.

    Returns:
        Summarized bills
    """
    # Embed query
    query_embedding = embed_model.encode(query).reshape(1, -1)

    # Use index to search bills
    _, indices = faiss_index.search(query_embedding, top_k * 5)

    # Get chunks
    retrieved_chunks = df.iloc[indices[0]]

    # Group by bill ID
    grouped = retrieved_chunks.groupby("measure_id").agg({
        "text_chunk": "\n".join,
        "title": "first",
        "orig_publish_date": "first",
        "action_date": "first",
        "action_desc": "first"
    }).reset_index()

    grouped = grouped.head(top_k)

    summaries = []

    for _, row in grouped.iterrows():
        bill_id = row["measure_id"]
        title = row["title"]
        orig_date = row["orig_publish_date"]
        action_date = row["action_date"]
        action_desc = row["action_desc"]
        full_text = row["text_chunk"]

        # Break large bill text into smaller chunks
        text_chunks = [full_text[i:i+1024] for i in range(0, len(full_text), 1024)]

        summarized_parts = []
        for chunk in text_chunks:
            tokens = tokenizer.encode(chunk, truncation=True)
            if len(tokens) < 30:
                continue  # Skip too-short content

            max_len = min(200, max(40, int(len(tokens) * 0.5)))  # Scale max length
            summary = summarizer(chunk, max_length=max_len, min_length=30, do_sample=False)[0]["summary_text"]
            summarized_parts.append(summary)

        if not summarized_parts:
            continue  # Skip if no meaningful summaries

        full_summary = " ".join(summarized_parts)

        formatted = f"""**Bill ID: {bill_id}**  
Title: {title}  
Origin Date: {orig_date}  
Action Date: {action_date}  
Action Description: {action_desc}  
Summary:  
{full_summary}"""
        
        summaries.append(formatted)
    
    clear_gpu_memory()  # Clear GPU memory after summarization
    return summaries


def chat_with_bills(query, top_k=5):
    """Retrieve the most relevant bills, group their chunks, summarize them,and generate a concise answer.

    Args:
        query (str): User query.
        top_k (int, optional): Retrieve top most relevant bills. Defaults to 5.

    Returns:
        LLM Response
    """
    retrieved_bills = retrieve_and_summarize_bills(query, top_k)

    if not isinstance(retrieved_bills, list):
        print("Error: Expected a list but got", type(retrieved_bills))
        return "I couldn't find relevant legislation."

    if not retrieved_bills:
        return "I couldn't find relevant legislation."

    # Remove duplicates based on the bill title
    unique_bills = {}
    for bill in retrieved_bills:
        title = bill.split("\n")[1]  # Assumes the second line is the title
        unique_bills[title] = bill  # Keep only the first occurrence of each title

    # Combine the preformatted bill summaries (no duplicates)
    formatted_text = "\n\n".join(unique_bills.values())

    prompt = f"""You are an expert on U.S. legislation. A user asked: "{query}"  
Here is relevant legislative information:

{formatted_text}

Please summarize the key points and answer the user’s question concisely. If no relevant text is found, say "I couldn't find relevant legislation."
"""

    # Set a smaller max_new_tokens if the input is large to avoid truncation
    response = llm(
        prompt,
        max_new_tokens=150,  # You can tweak this based on the length of your prompt
        truncation=True,
        do_sample=True
    )[0]["generated_text"]
    
    # clear_gpu_memory()  # Clear GPU memory after summarization if needed
    return response


if __name__ == "__main__":
    query = "What is the Fentanyl Act?"
    response = chat_with_bills(query, top_k=5)
    print(response)
