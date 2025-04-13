import streamlit as st
from rag_llm_inference import rag, load_model

@st.cache_resource
def load_default_model():
    return load_model("Qwen/Qwen2.5-7B-Instruct-1M")

tokenizer, model = load_default_model()

with st.sidebar:
    "[View the source code](https://github.com/sruxll/rag_us_congressional_bills)"

st.title("U.S. Congressional Bills Chatbot 💬")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = rag(prompt, tokenizer, model)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
