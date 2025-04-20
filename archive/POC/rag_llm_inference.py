from marqo import Client
import numpy as np
import csv
# Load model directly from huggingface
from transformers import AutoTokenizer, AutoModelForCausalLM


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
<answer> your final concise answer </answer>
<explanation> Your well structured, citation rich answer that follows the rules above </explanation>
"""


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def run_llm_inference(prompt, tokenizer, model):
    in_tensor = tokenizer(prompt, return_tensors="pt")
    out_tensor = model.generate(**in_tensor, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(in_tensor.input_ids, out_tensor)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def rag(prompt, tokenizer, model):
    # connect to the existing Marqo index
    mq = Client('http://localhost:8882')
    index_name = 'text-search-bills'

    # define the query and search the index
    query = prompt
    results = mq.index(index_name).search(query)

    # build context with up to 3 documents retrieved
    contexts = [
        f"[{results['hits'][i]['measure_id']}] {results['hits'][i]['text_chunk']}"
        for i in range(min(3, len(results['hits'])))
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

