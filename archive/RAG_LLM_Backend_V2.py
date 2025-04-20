import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import ast  # To safely evaluate string representations of lists
from tqdm import tqdm
from nltk.corpus import wordnet
from huggingface_hub import snapshot_download
import torch
import re
"""Make sure to upload "parsed_bills_115-119_chunks_only_embedded.csv and "bill_embeddings.index in working directory"""


# Load Sentence Transformer for embeddings
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda")

# Initialize tokenizer for summarizer model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Load Summarization Model
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device="cuda",                # Use the first available GPU (cuda), or CPU
    torch_dtype="auto",      # Automatically switch to float16 (efficient)
    max_length=150,          # Control max output length
    min_length=30            # Control minimum output length
)

# Load FAISS Index
faiss_index = faiss.read_index("bill_embeddings.index")

# Load Data
file_path = "parsed_bills_115-119_chunks_only_embedded.csv"
df = pd.read_csv(file_path)

# Convert embeddings from CSV to NumPy Arrays
df["embedding"] = df["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))

# Load in Text Generation Model
model_llm_id = "tiiuae/falcon-rw-1b"

tokenizer_llm = AutoTokenizer.from_pretrained(model_llm_id)
model_llm = AutoModelForCausalLM.from_pretrained(model_llm_id)
llm = pipeline("text-generation",
               model=model_llm,
               tokenizer=tokenizer_llm,
               device="cuda")


def retrieve_and_summarize_bills(query, top_k=3):
    """
    Retrieves and summarizes the most relevant U.S. legislative bills based on a user query.

    Args:
        query (str): The user's search question or topic of interest.
        top_k (int): Number of top distinct bills to return after grouping and summarization.

    Returns:
        bullet_summaries (List[str]): List of formatted summaries with bill titles and summaries.
        grouped_ids (Dict[str, List[str]]): Mapping of bill titles to their associated bill IDs.
    """
    # Convert the user query into an embedding
    query_embedding = embed_model.encode(query).reshape(1, -1)

    # Perform FAISS search to get top candidate chunks (more than top_k to allow grouping)
    _, indices = faiss_index.search(query_embedding, top_k * 10)

    # Get the corresponding chunks and build a composite key for grouping
    retrieved_chunks = df.iloc[indices[0]].copy()
    retrieved_chunks["group_key"] = retrieved_chunks["title"] + " | " + retrieved_chunks["measure_id"]

    # Group retrieved chunks by bill title, keeping up to 5 chunks per bill
    summary_groups = (
        retrieved_chunks
        .groupby("title", group_keys=False)
        .apply(
            lambda group: pd.Series({
                "combined_text": "\n".join(group["text_chunk"].values[:5]),  # First 5 chunks only
                "bill_ids": list(group["measure_id"].unique())  # All unique bill IDs per title
            }),
            include_groups=False
        )
        .reset_index()
        .head(top_k)  # Limit to top_k grouped bills
    )

    bullet_summaries = []
    grouped_ids = {}

    for _, row in summary_groups.iterrows():
        title = row["title"]
        snippet = row["combined_text"][:1000]  # Token-efficient truncation of text input

        # Prompt LLM to generate a clean, brief summary without repeating the title
        prompt = f"""You are a U.S. legislative analyst. Write a clear and concise 1–2 sentence summary of the following bill's contents. Do NOT mention the bill's title or say "The bill is titled". Just summarize what the bill does.

--- Bill Details ---
{snippet}

Summary:"""

        # Generate summary using LLM
        response = llm(
            prompt,
            max_new_tokens=300,
            do_sample=False
        )[0]["generated_text"]

        # Extract only the generated portion after the prompt
        answer = response[len(prompt):].strip()

        # Remove any unwanted repeated phrases
        answer = re.sub(r"^(The bill is titled.*?)($|\s)", "", answer, flags=re.IGNORECASE).strip()

        # Truncate to 2 sentences max for clarity
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        clean_summary = " ".join(sentences[:2]).strip()

        # Fallback if summary is too short or empty
        if not clean_summary or len(clean_summary.split()) < 5:
            clean_summary = "Summary not available for this bill."

        # Format output as a markdown bullet summary
        bill_id = row['bill_ids'][0]
        bullet_summaries.append(f"**{title} ({bill_id})**: {clean_summary}")
        grouped_ids[title] = row["bill_ids"]

    return bullet_summaries, grouped_ids


def chat_with_bills(query, top_k=3):
    """
    Generates a user-facing response containing summarized legislative bills relevant to the query.

    Args:
        query (str): The user's natural-language query about legislation.
        top_k (int): Number of top distinct bills to display.

    Returns:
        str: Combined markdown-style summaries and list of top relevant bills.
    """
    summaries, grouped_ids = retrieve_and_summarize_bills(query, top_k)

    if not summaries:
        return "Sorry, I couldn't find any relevant legislation."

    # Join the summaries and prepare a readable list of top matched bills
    summary_section = "\n\n".join(summaries)

    top_bill_list = "\n".join([
        f"- {title} ({', '.join(ids)})" for title, ids in grouped_ids.items()
    ])

    return f"{summary_section}\n\nTop Relevant Bills:\n{top_bill_list}"


if __name__ == "__main__":
    query = "Are there any active or proposed bills for student loan forgiveness?"
    print(chat_with_bills(query, top_k=5))
