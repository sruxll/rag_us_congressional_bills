import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import ast  # To safely evaluate string representations of lists
import torch
import re
import csv
from bert_score import score as bert_score

# === Load Models and Resources ===
embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu"
)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype="auto",
    max_length=150,
    min_length=30,
)
model_llm_id = "tiiuae/falcon-rw-1b"
tokenizer_llm = AutoTokenizer.from_pretrained(model_llm_id)
model_llm = AutoModelForCausalLM.from_pretrained(model_llm_id)
llm = pipeline(
    "text-generation",
    model=model_llm,
    tokenizer=tokenizer_llm,
    device=0 if torch.cuda.is_available() else -1,
)

# === Load FAISS Index and Data ===
faiss_index = faiss.read_index("bill_embeddings.index")

# Download the CSV from Hugging Face Hub
csv_path = hf_hub_download(
    repo_id="joynae5/CongressionalBillsDS",
    filename="data/parsed_bills_115-119_chunks_only_embedded.csv",
    repo_type="dataset",
)

df = pd.read_csv(csv_path)
df["embedding"] = df["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))

# === Similarity Tracking ===
similarity_scores = []
bert_scores = []
similarity_saved_path = "similarity_scores.csv"
with open(similarity_saved_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Title", "ID", "Cosine Similarity", "BERTScore", "Flagged?"])


def retrieve_and_summarize_bills(query, top_k=3):
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

        prompt = f"""You are a U.S. legislative analyst. Write a clear and concise 1–2 sentence summary of the following bill's contents. Do NOT mention the bill's title or say \"The bill is titled\". Just summarize what the bill does.

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

        summary_emb = embed_model.encode(clean_summary, convert_to_tensor=True)
        snippet_emb = embed_model.encode(snippet, convert_to_tensor=True)
        similarity = util.cos_sim(summary_emb, snippet_emb).item()
        similarity_scores.append(similarity)

        P, R, F1 = bert_score(
            [clean_summary], [snippet], lang="en", rescale_with_baseline=True
        )
        f1_score_bert = F1[0].item()
        bert_scores.append(f1_score_bert)

        risky = similarity < 0.6 or f1_score_bert < 0.85
        risk_flag = (
            "\n⚠️ Possible hallucination (low similarity or BERTScore)" if risky else ""
        )
        is_similar = "Yes" if risky else "No"

        bill_id = row["bill_ids"][0]
        bullet_summaries.append(f"**{title} ({bill_id})**: {clean_summary}{risk_flag}")
        grouped_ids[title] = row["bill_ids"]

        with open(similarity_saved_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    title,
                    bill_id,
                    round(similarity, 4),
                    round(f1_score_bert, 4),
                    is_similar,
                ]
            )

    return bullet_summaries, grouped_ids


def chat_with_bills(query, top_k=3):
    summaries, grouped_ids = retrieve_and_summarize_bills(query, top_k)

    if not summaries:
        return "Sorry, I couldn't find any relevant legislation."

    summary_section = "\n\n".join(summaries)
    top_bill_list = "\n".join(
        [f"- {title} ({', '.join(ids)})" for title, ids in grouped_ids.items()]
    )

    return f"{summary_section}\n\nTop Relevant Bills:\n{top_bill_list}"


if __name__ == "__main__":
    query = "Are there any active or proposed bills for student loan forgiveness?"
    print(chat_with_bills(query, top_k=5))
