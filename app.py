import os
import glob
import streamlit as st

st.set_page_config(page_title="Pergunte ao PPA", page_icon="")

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader
)

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# 🔐 API
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 📁 Diretórios
DOCS_PATH = "./docs"
VECTORDB_PATH = "./vectordb"
os.makedirs(DOCS_PATH, exist_ok=True)
os.makedirs(VECTORDB_PATH, exist_ok=True)

# ✅ LLM
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=openai_key
    )

# ✅ Embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# 📚 Cria a base FAISS
def create_vectorstore():
    st.info("🔄 Reindexando documentos...")
    docs = []
    files = sorted(glob.glob(f"{DOCS_PATH}/*"))

    # Barra de progresso
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    total = len(files)

    for i, file in enumerate(files):
        ext = os.path.splitext(file)[1].lower()
        filename = os.path.basename(file)
        status_placeholder.markdown(f"📄 Processando: `{filename}`")

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file)
            elif ext == ".txt":
                loader = TextLoader(file)
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(file)
            elif ext == ".xlsx":
                loader = UnstructuredExcelLoader(file)
            elif ext == ".html":
                loader = UnstructuredHTMLLoader(file)
            else:
                continue

            docs.extend(loader.load())

        except Exception as e:
            st.warning(f"⚠️ Erro ao processar `{filename}`: {e}")

        # Atualiza barra de progresso
        progress_bar.progress((i + 1) / total)

    if not docs:
        st.error("❌ Nenhum documento válido encontrado em ./docs")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = load_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTORDB_PATH)

    st.session_state["indexed_files"] = [os.path.basename(f) for f in files if os.path.isfile(f)]

    status_placeholder.success("✅ Documentos indexados com sucesso.")
    progress_bar.empty()
    return db


# ✅ Carrega FAISS (sem auto verificação)
def load_vectorstore():
    if not os.path.exists(os.path.join(VECTORDB_PATH, "index.faiss")):
        st.warning("⚠️ Nenhuma base vetorial encontrada. Clique em '🔁 Reindexar agora'.")
        return None

    embeddings = load_embeddings()
    files = sorted(glob.glob(f"{DOCS_PATH}/*"))
    st.session_state["indexed_files"] = [os.path.basename(f) for f in files if os.path.isfile(f)]
    return FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)

# 📤 Upload
# 📤 Upload
st.sidebar.header("📤 Enviar documentos")
uploaded_files = st.sidebar.file_uploader(
    "Arquivos permitidos: .pdf, .txt, .docx, .xlsx, .html",
    type=["pdf", "txt", "docx", "xlsx", "html"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(DOCS_PATH, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
    st.sidebar.success("✅ Arquivos enviados com sucesso.")

# 🔘 Reindexar manualmente
if st.sidebar.button("🔁 Reindexar agora"):
    create_vectorstore()
    st.rerun()

# 📂 Arquivos indexados
st.sidebar.markdown("📂 **Arquivos indexados:**")
if "indexed_files" in st.session_state and st.session_state["indexed_files"]:
    for f in st.session_state["indexed_files"]:
        st.sidebar.markdown(f"- `{f}`")
else:
    st.sidebar.info("Nenhum arquivo indexado ainda.")


# 🚀 Inicializa o LLM
llm = load_llm()
vectorstore = load_vectorstore()

# 🔁 RAG Chain
if vectorstore:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", k=4),
        return_source_documents=True
    )
else:
    qa_chain = None

# 🧠 Interface
st.title("Pergunte ao PPA")


# 💬 Histórico
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 📝 Entrada
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

# 💬 Mostrar histórico
for role, msg in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(msg)

# 📄 Fontes usadas
if "last_contexts" in st.session_state:
    with st.expander("📚 Trechos usados na resposta"):
        for doc in st.session_state.last_contexts:
            nome = os.path.basename(doc.metadata.get("source", ""))
            st.markdown(f"**Fonte:** `{nome}`")
            st.markdown(doc.page_content.strip())
            st.markdown("---")

# 🧹 Limpar conversa
if st.button("🧹 Limpar conversa"):
    st.session_state.chat_history = []
    st.session_state.last_contexts = []
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# 💾 Download
if st.session_state.chat_history:
    for role, msg in reversed(st.session_state.chat_history):
        if role == "bot":
            st.download_button("📥 Baixar última resposta", msg, file_name="resposta.txt")
            break
