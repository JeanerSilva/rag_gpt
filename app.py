import os
import streamlit as st
from config import DOCS_PATH, load_llm
from rag.vectorstore import load_vectorstore, create_vectorstore
from rag.qa_chain import build_qa_chain
from rag.utils import save_uploaded_files, load_indexed_files

# Setup inicial
st.set_page_config(page_title="Pergunte ao PPA", page_icon="")
st.title("Pergunte ao PPA")

# Session state
if "indexed_files" not in st.session_state:
    st.session_state["indexed_files"] = load_indexed_files()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload

# ⚙️ Configuração dinâmica do 'k'
st.sidebar.markdown("⚙️ **Configurações**")
k_value = st.sidebar.number_input(
    label="Número de trechos a considerar (k)",
    min_value=1,
    max_value=20,
    value=st.session_state.get("retriever_k", 6),
    step=1,
    key="retriever_k"
)


st.sidebar.header("📤 Enviar documentos")
uploaded_files = st.sidebar.file_uploader(
    "Arquivos: .pdf, .txt, .docx, .xlsx, .html",
    type=["pdf", "txt", "docx", "xlsx", "html"],
    accept_multiple_files=True,
)
if uploaded_files:
    save_uploaded_files(uploaded_files)
    st.sidebar.success("✅ Arquivos enviados com sucesso.")

# Reindexar
if st.sidebar.button("🔁 Reindexar agora"):
    create_vectorstore()

# Lista compacta de arquivos indexados
indexed_files = st.session_state.get("indexed_files", [])
if indexed_files:
    st.sidebar.markdown("📂 **Arquivos indexados:**", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<ul style='padding-left:1.2em;'>"
        + "".join(
            f"<li style='font-size:0.8em; margin-bottom:0.1em;'>{f}</li>"
            for f in indexed_files
        )
        + "</ul>",
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown("📂 Nenhum arquivo indexado.")

# RAG
llm = load_llm()
vectorstore = load_vectorstore()
qa_chain = build_qa_chain(llm, vectorstore)

# Pergunta do usuário
if qa_chain:
    with st.form("chat-form", clear_on_submit=True):
        user_input = st.text_input("Digite sua pergunta:")
        submitted = st.form_submit_button("Enviar")
    if submitted and user_input:
        result = qa_chain(user_input)
        resposta = result["result"]
        fontes = result["source_documents"]

        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", resposta))
        st.session_state.last_contexts = fontes

# Histórico
for role, msg in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(msg)

# Fontes usadas
if "last_contexts" in st.session_state:
    with st.expander("📚 Trechos usados na resposta"):
        for doc in st.session_state.last_contexts:
            nome = os.path.basename(doc.metadata.get("source", ""))
            st.markdown(f"**Fonte:** `{nome}`")
            st.markdown(doc.page_content.strip())
            st.markdown("---")

# Limpar
if st.button("🧹 Limpar conversa"):
    st.session_state.chat_history = []
    st.session_state.last_contexts = []
    st.rerun()

# Baixar
if st.session_state.chat_history:
    for role, msg in reversed(st.session_state.chat_history):
        if role == "bot":
            st.download_button("📥 Baixar última resposta", msg, file_name="resposta.txt")
            break
