import os
import glob
import hashlib
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

# 🔐 Carrega a chave da API
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

# 🔍 Hash dos arquivos
def hash_file_list(folder):
    files = sorted(glob.glob(f"{folder}/*"))
    hash_input = "".join(
        f"{os.path.basename(f)}-{os.path.getmtime(f)}"
        for f in files if os.path.isfile(f)
    )
    return hashlib.md5(hash_input.encode()).hexdigest()

# 📚 Cria a base FAISS
def create_vectorstore():
    st.info("🔄 Reindexando documentos...")
    docs = []
    files = glob.glob(f"{DOCS_PATH}/*")

    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file)
        elif ext == ".txt":
            loader = TextLoader(file)
        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(file)
        elif ext == ".xlsx":
            loader = UnstructuredExcelLoader(file)
        elif ext == ".html":
            loader = UnstructuredHTMLLoader(file)  # 👈 SUPORTE HTML
        else:
            continue
        docs.extend(loader.load())


    if not docs:
        st.error("❌ Nenhum documento válido encontrado em ./docs")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = load_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTORDB_PATH)
    return db

# ✅ Carrega ou recria FAISS
def load_vectorstore():
    current_hash = hash_file_list(DOCS_PATH)
    files = sorted(glob.glob(f"{DOCS_PATH}/*"))
    file_names = [os.path.basename(f) for f in files if os.path.isfile(f)]

    force = st.session_state.pop("force_reindex", False)

    hash_changed = (
        "last_docs_hash" not in st.session_state or
        st.session_state["last_docs_hash"] != current_hash or
        not os.path.exists(os.path.join(VECTORDB_PATH, "index.faiss")) or
        force  # 👈 força manual
    )

    if hash_changed:
        st.session_state["last_docs_hash"] = current_hash
        st.session_state["indexed_files"] = file_names
        return create_vectorstore()

    if "indexed_files" not in st.session_state:
        st.session_state["indexed_files"] = file_names

    embeddings = load_embeddings()
    return FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)


    if "indexed_files" not in st.session_state:
        st.session_state["indexed_files"] = file_names

    embeddings = load_embeddings()
    return FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)

# 📤 Upload pela sidebar
st.sidebar.header("📤 Enviar documentos")
uploaded_files = st.sidebar.file_uploader(
    "Escolha arquivos (.pdf, .txt, .docx, .xlsx, .html)",
    type=["pdf", "txt", "docx", "xlsx", "html"],
    accept_multiple_files=True
)


if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(DOCS_PATH, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
    st.sidebar.success("✅ Arquivos enviados.")
    if "last_docs_hash" in st.session_state:
        del st.session_state["last_docs_hash"]
    st.rerun()

# 🔘 Botão para reindexar manualmente
if st.sidebar.button("🔁 Reindexar agora"):
    # Remove o hash anterior para forçar a reindexação
    if "last_docs_hash" in st.session_state:
        del st.session_state["last_docs_hash"]

    # Atualiza lista de arquivos
    files = sorted(glob.glob(f"{DOCS_PATH}/*"))
    file_names = [os.path.basename(f) for f in files if os.path.isfile(f)]
    st.session_state["indexed_files"] = file_names

    # Força mensagem de reindexação via variável de estado
    st.session_state["force_reindex"] = True

    # Recarrega o app
    st.rerun()


# 🚀 Inicializa
llm = load_llm()
vectorstore = load_vectorstore()

# 🔁 Chain com retorno de fontes
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_type="similarity", k=4),
    return_source_documents=True
)

# 🧠 Interface principal
st.title("🧠 Chat com seus Documentos (RAG + GPT)")

# 📂 Lista de arquivos indexados
if "indexed_files" in st.session_state and st.session_state["indexed_files"]:
    st.markdown("📁 **Arquivos indexados:**")
    for f in st.session_state["indexed_files"]:
        st.markdown(f"- `{f}`")
else:
    st.warning("⚠️ Nenhum arquivo foi indexado ainda.")

# 💬 Histórico
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 📝 Entrada do usuário
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

# 📄 Mostrar fontes usadas
if "last_contexts" in st.session_state:
    with st.expander("📚 Trechos usados na resposta"):
        for doc in st.session_state.last_contexts:
            nome = os.path.basename(doc.metadata.get("source", ""))
            st.markdown(f"**Fonte:** `{nome}`")
            st.markdown(doc.page_content.strip())
            st.markdown("---")

# 🧹 Botão limpar
if st.button("🧹 Limpar conversa"):
    st.session_state.chat_history = []
    st.session_state.last_contexts = []
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


# 💾 Download da última resposta
if st.session_state.chat_history:
    for role, msg in reversed(st.session_state.chat_history):
        if role == "bot":
            st.download_button("📥 Baixar última resposta", msg, file_name="resposta.txt")
            break
