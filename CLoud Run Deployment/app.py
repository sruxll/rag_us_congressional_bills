import os
import re
import streamlit as st
from RAG_embedding_LLMgeneration import rag, load_files
from google import genai


@st.cache_resource
def read_files():
    return load_files()


client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
faiss_index, df = read_files()

with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/sruxll/rag_us_congressional_bills/main/serving-rag-gemini/us_congress_image.jpg",
        caption="",
        use_container_width=True,
    )
    st.markdown(
        """
    ## Capitol Insight
    This chatbot lets you ask questions about U.S. Congressional bills from the 115th to 119th Congress.

    **Instructions:**
    - Type a question below.
    - The chatbot will retrieve relevant bill text and provide an answer.
    - Document IDs in parentheses let you look up full bills on [GovInfo](https://www.govinfo.gov/app/collection/BILLS).

    [View the source code in GitHub](https://github.com/sruxll/rag_us_congressional_bills)
    """
    )

st.title("U.S. Congressional Bills Chatbot 💬")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=False)

if prompt := st.chat_input():
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=False)

    msg = rag(prompt, client, faiss_index, df)

    answer_match = re.search(r"<answer>(.*?)</answer>", msg, re.DOTALL)
    explain_match = re.search(r"<explanation>(.*?)</explanation>", msg, re.DOTALL)
    answer_text = answer_match.group(1).strip() if answer_match else ""
    explain_text = explain_match.group(1).strip() if explain_match else ""

    combined_message = (
        f"**Answer:**\n\n{answer_text}\n\n**Explanation:**\n\n{explain_text}"
        if answer_text and explain_text
        else msg
    )

    def inline_citations(text):
        text = re.sub(r"\[(id[\w\d\s,]+)\]", r"(\1)", text)
        text = re.sub(r"\{(id[\w\d\s,]+)\}", r"(\1)", text)
        return text

    combined_message = inline_citations(combined_message)

    st.session_state["messages"].append(
        {"role": "assistant", "content": combined_message}
    )
    with st.chat_message("assistant"):
        st.markdown(combined_message, unsafe_allow_html=False)
