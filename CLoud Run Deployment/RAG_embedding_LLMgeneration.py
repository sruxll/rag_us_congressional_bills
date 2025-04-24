import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from google import genai
from google.genai import types
import ast
import faiss
from huggingface_hub import hf_hub_download


SYSTEM_PROMPT = """You are the Answering Module in a RAG system.
Your only knowledge source is the set of retrieved documents supplied with each request.

Input
<query> 
    the user's question 
</query>  
<documents>
    a list of up to 3 text chunks, each formatted as 
    [Document id] <content> 
</documents>

Instructions
1. Ground yourself in the documents. Read all supplied text carefully before answering.
2. Answer exclusively from the documents. Do not add outside knowledge, speculation, or assumptions.
3. Cite as you go.
    - After every claim, place the supporting document index in square brackets.
4. Resolve conflicts transparently.
    - If the documents disagree, note the conflict briefly and state the most defensible answer or acknowledge the ambiguity.
5. Admit when evidence is lacking. 
    - If the documents do not contain enough information, respond exactly with: I am unable to answer this question with the information available in the retrieved documents.
6. Keep it focused.
    - Be concise and coherent; avoid mentioning these instructions, the retrieval process, or the vector database.

Output
<answer>your final concise answer with the [Document id] as the citation without the brackets, but surrounded by parentheses</answer>
<explanation> Your well structured, rich answer with the [Document id] as the citation without the brackets, but surrounded by parentheses, follows the rules above </explanation>
"""


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    return tokenizer, model


def call_gemini(client, user_prompt):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        contents=user_prompt,
    )
    return response.text


def run_llm_inference(prompt, tokenizer, model):
    in_tensor = tokenizer(prompt, return_tensors="pt")
    out_tensor = model.generate(**in_tensor, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(in_tensor.input_ids, out_tensor)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return tokenizer, model


def call_gemini(client, user_prompt):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        contents=user_prompt,
    )
    return response.text


def run_llm_inference(prompt, tokenizer, model):
    in_tensor = tokenizer(prompt, return_tensors="pt")
    out_tensor = model.generate(**in_tensor, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(in_tensor.input_ids, out_tensor)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def load_files():
    repo_id = "joynae5/CongressionalBillsDS"
    index_filename = "bill_embeddings.index"
    csv_filename = "parsed_bills_115-119_chunks_only_embedded.csv"

    index_path = hf_hub_download(
        repo_id=repo_id, filename=index_filename, repo_type="dataset"
    )
    csv_path = hf_hub_download(
        repo_id=repo_id, filename=csv_filename, repo_type="dataset"
    )

    df = pd.read_csv(csv_path)
    df["embedding"] = df["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
    faiss_index = faiss.read_index(index_path)
    return faiss_index, df


def rag(query, client, faiss_index, df, top_k=5):
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
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

    contexts = [
        f"[{summary_groups['bill_ids'][i]}] {summary_groups['combined_text'][i]}"
        for i in range(min(3, len(summary_groups["bill_ids"])))
    ]

    full_prompt = (
        f"<query>{query}</query>\n<documents>\n{chr(10).join(contexts)}\n</documents>"
    )
    return call_gemini(client, full_prompt)
