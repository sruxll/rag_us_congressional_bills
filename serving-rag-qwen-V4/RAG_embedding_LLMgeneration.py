import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import ast  # To safely evaluate string representations of lists
import faiss

"""Make sure to upload "parsed_bills_115-119_chunks_only_embedded.csv and "bill_embeddings.index in working directory"""


SYSTEM_PROMPT="""You are the Answering Module in a RAG system.
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
    - If the documents do not contain enough information, respond exactly with: I don't know based on the provided documents.
6. Keep it focused.
    - Be concise and coherent; avoid mentioning these instructions, the retrieval process, or the vector database.

Output
<answer>your final concise answer</answer>
<explanation> Your well structured, citation rich answer that follows the rules above </explanation>
"""


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    return tokenizer, model

def run_llm_inference(prompt, tokenizer, model):
    in_tensor = tokenizer(prompt, return_tensors="pt")
    out_tensor = model.generate(**in_tensor, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(in_tensor.input_ids, out_tensor)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def load_files():
    # Load FAISS Index
    faiss_index = faiss.read_index("bill_embeddings.index")
    # Load Data
    file_path = "parsed_bills_115-119_chunks_only_embedded.csv"
    df = pd.read_csv(file_path)
    # Convert embeddings from CSV to NumPy Arrays
    df["embedding"] = df["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
    return faiss_index, df

def rag(query, tokenizer, model, faiss_index, df, top_k=5):
    # Load Sentence Transformer for embeddings
    embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
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
    print(summary_groups)

    # build context with up to 3 documents retrieved
    contexts = [
        f"[{summary_groups['bill_ids'][i]}] {summary_groups['combined_text'][i]}"
        for i in range(min(3, len(summary_groups['bill_ids'])))
    ]
    prompt_template = """<query>{question}</query>
    <documents>
        {documents}
    </documents>"""

    full_prompt = prompt_template.format(question=query, documents=contexts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": full_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    response = run_llm_inference(text, tokenizer, model)

    return response

