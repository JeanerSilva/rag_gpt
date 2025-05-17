from langchain.chains import RetrievalQA
from rag.prompt import get_custom_prompt
from settings import RETRIEVER_TOP_K

import streamlit as st

def build_qa_chain(llm, vectorstore):
    if not vectorstore:
        return None
    k_value = st.session_state.get("retriever_k", 6)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", k=k_value),
        return_source_documents=True,
        chain_type_kwargs={"prompt": get_custom_prompt()}
    )