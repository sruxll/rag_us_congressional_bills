import re
import streamlit as st
from RAG_embedding_LLMgeneration import rag, load_model

@st.cache_resource
def load_default_model():
    return load_model("Qwen/Qwen2.5-7B-Instruct-1M")

tokenizer, model = load_default_model()

with st.sidebar:
    # add an image
    st.image("us_congress_image.jpg", caption="", use_container_width=True)    
    "[View the source code in GitHub](https://github.com/sruxll/rag_us_congressional_bills)"

st.title("U.S. Congressional Bills Chatbot 💬")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state["messages"]:
    if msg["role"] == "assistant":
        st.chat_message("assistant").markdown(msg["content"])
    else:
        st.chat_message("user").markdown(msg["content"])


if prompt := st.chat_input():
    # append and display user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    msg = rag(prompt, tokenizer, model)
    
    # extract answer and explanation
    answer_match = re.search(r'<answer>(.*?)</answer>', msg, re.DOTALL)
    explain_match = re.search(r'<explanation>(.*?)</explanation>', msg, re.DOTALL)
    answer_text = answer_match.group(1).strip() if answer_match else ""
    explain_text = explain_match.group(1).strip() if explain_match else ""
    
    # format markdown message
    combined_message = (
        f"**Answer:**\n\n{answer_text}\n\n"
        f"**Explanation:**\n\n{explain_text}"
        if answer_text and explain_text
        else msg
    )
    
    # display assistant's response
    st.session_state["messages"].append({"role": "assistant", "content": combined_message})
    st.chat_message("assistant").markdown(combined_message)
