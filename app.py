import os
import glob
import streamlit as st

st.set_page_config(page_title="Pergunte ao PPA", page_icon="")

import json

INDEXED_LIST_PATH = "indexed_files.json"

def save_indexed_files(file_list):
    with open(INDEXED_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(file_list, f, ensure_ascii=False, indent=2)

def load_indexed_files():
    if os.path.exists(INDEXED_LIST_PATH):
        with open(INDEXED_LIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

if "indexed_files" not in st.session_state:
    st.session_state["indexed_files"] = load_indexed_files()

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
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

import json

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
    sidebar_status = st.sidebar.empty()
    sidebar_progress = st.sidebar.progress(0)

    sidebar_status.info("🔄 Reindexando documentos...")
    docs = []
    files = sorted(glob.glob(f"{DOCS_PATH}/*"))
    total = len(files)

    for i, file in enumerate(files):
        ext = os.path.splitext(file)[1].lower()
        filename = os.path.basename(file)
        sidebar_status.markdown(f"📄 Processando: `{filename}`")

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
            st.sidebar.warning(f"⚠️ Erro ao processar `{filename}`: {e}")

        sidebar_progress.progress((i + 1) / total)

    if not docs:
        sidebar_status.error("❌ Nenhum documento válido encontrado.")
        sidebar_progress.empty()
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = load_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTORDB_PATH)

    st.session_state["indexed_files"] = [os.path.basename(f) for f in files if os.path.isfile(f)]

    indexed_list = [os.path.basename(f) for f in files if os.path.isfile(f)]
    st.session_state["indexed_files"] = indexed_list
    save_indexed_files(indexed_list)  # 👈 salvar em disco

    sidebar_status.success("✅ Documentos indexados com sucesso.")
    sidebar_progress.empty()
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


# 📂 Arquivos indexados
indexed_files = st.session_state.get("indexed_files", [])

if indexed_files:
    st.sidebar.markdown("📂 **Arquivos indexados:**", unsafe_allow_html=True)
    styled_list = "<ul style='padding-left: 1.2em; margin-top: 0.2em;'>"
    for f in indexed_files:
        styled_list += f"<li style='font-size: 0.8em; margin-bottom: 0.1em;'>{f}</li>"
    styled_list += "</ul>"
    st.sidebar.markdown(styled_list, unsafe_allow_html=True)
else:
    st.sidebar.markdown("📂 Nenhum arquivo indexado.")


# 🚀 Inicializa o LLM
llm = load_llm()
vectorstore = load_vectorstore()


# Prompt 
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Você é um assistente especializado em planejamento público e está respondendo a uma pergunta com base nos trechos de documentos oficiais abaixo.

🔹 **Contexto** (extraído dos documentos indexados):
{context}

🔹 **Pergunta do usuário:**
{question}

💡 **Instruções**:
- Responda de forma clara, objetiva e embasada.
- Utilize o conteúdo dos documentos como referência principal.
- Se necessário, cite trechos ou dados para justificar sua resposta.
- Se os documentos não contiverem informações suficientes, informe isso de forma clara — mas não invente.

📝 **Resposta**:
"""
)

# 🔁 RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_type="similarity", k=6),
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

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

