import os
import re
import ast
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from nltk.corpus import wordnet
from huggingface_hub import hf_hub_download
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss

# download ds from huggingface
index_path = hf_hub_download(
    repo_id="joynae5/CongressionalBillsDS", filename="bill_embeddings.index"
)

csv_path = hf_hub_download(
    repo_id="joynae5/CongressionalBillsDS",
    filename="parsed_bills_115-119_chunks_only_embedded.csv",
)

# load model
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# BART
tokenizer_summarizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device="cuda",
    torch_dtype="auto",
    max_length=150,
    min_length=30,
)

# Text Generator
model_llm_id = "tiiuae/falcon-rw-1b"
tokenizer_llm = AutoTokenizer.from_pretrained(model_llm_id)
model_llm = AutoModelForCausalLM.from_pretrained(model_llm_id)
llm = pipeline(
    "text-generation", model=model_llm, tokenizer=tokenizer_llm, device="cuda"
)

# load ds
faiss_index = faiss.read_index(index_path)

df = pd.read_csv(csv_path)
df["embedding"] = df["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))


def retrieve_and_summarize_bills(query, top_k=3):
    """
    Retrieves and summarizes relevant U.S. bills based on a query.
    """
    query_embedding = embed_model.encode(query).reshape(1, -1)
    _, indices = faiss_index.search(query_embedding, top_k * 10)

    retrieved_chunks = df.iloc[indices[0]].copy()
    retrieved_chunks["group_key"] = (
        retrieved_chunks["title"] + " | " + retrieved_chunks["measure_id"]
    )

    summary_groups = (
        retrieved_chunks.groupby("title", group_keys=False)
        .apply(
            lambda group: pd.Series(
                {
                    "combined_text": "\n".join(group["text_chunk"].values[:5]),
                    "bill_ids": list(group["measure_id"].unique()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .head(top_k)
    )

    bullet_summaries = []
    grouped_ids = {}

    for _, row in summary_groups.iterrows():
        title = row["title"]
        snippet = row["combined_text"][:1000]

        prompt = f"""You are a U.S. legislative analyst. Write a clear and concise 1–2 sentence summary of the following bill's contents. Do NOT mention the bill's title or say "The bill is titled". Just summarize what the bill does.

--- Bill Details ---
{snippet}

Summary:"""

        response = llm(prompt, max_new_tokens=300, do_sample=False)[0]["generated_text"]
        answer = response[len(prompt) :].strip()
        answer = re.sub(
            r"^(The bill is titled.*?)($|\s)", "", answer, flags=re.IGNORECASE
        ).strip()
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        clean_summary = " ".join(sentences[:2]).strip()

        if not clean_summary or len(clean_summary.split()) < 5:
            clean_summary = "Summary not available for this bill."

        bill_id = row["bill_ids"][0]
        bullet_summaries.append(f"**{title} ({bill_id})**: {clean_summary}")
        grouped_ids[title] = row["bill_ids"]

    return bullet_summaries, grouped_ids


# chat portion
def chat_with_bills(query, top_k=3):
    """
    Returns formatted summaries of top relevant bills for a user query.
    """
    summaries, grouped_ids = retrieve_and_summarize_bills(query, top_k)

    if not summaries:
        return "Sorry, I couldn't find any relevant legislation."

    summary_section = "\n\n".join(summaries)
    top_bill_list = "\n".join(
        [f"- {title} ({', '.join(ids)})" for title, ids in grouped_ids.items()]
    )

    return f"{summary_section}\n\nTop Relevant Bills:\n{top_bill_list}"


# User Input added
if __name__ == "__main__":
    print("U.S. Congressional Bill Summarizer ")
    user_query = input("Enter your question: ").strip()

    if not user_query:
        print("No query provided. Please enter a topic or question.")
    else:
        print("\nSearching and summarizing relevant bills...\n")
        result = chat_with_bills(user_query, top_k=5)
        print(result)
