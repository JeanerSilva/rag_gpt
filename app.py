import os
import glob
import streamlit as st

# ✅ PRIMEIRO COMANDO do Streamlit — OBRIGATÓRIO
st.set_page_config(page_title="Chat RAG GPT", page_icon="🧠")

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# 🔐 Carrega variáveis do .env (API Key)
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 🔧 Configurações de caminho
DOCS_PATH = "./docs"
VECTORDB_PATH = "./vectordb"

# ✅ LLM OpenAI
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=openai_key
    )

# ✅ Embeddings locais (modelo leve)
@st.cache_resource
def load_embeddings():
    return HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# ✅ Cria base vetorial se não existir
@st.cache_resource
def create_vectorstore():
    st.info("🔄 Criando base vetorial a partir dos documentos...")

    docs = []
    files = glob.glob(f"{DOCS_PATH}/*")

    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file)
        elif ext == ".txt":
            loader = TextLoader(file)
        else:
            continue
        docs.extend(loader.load())

    if not docs:
        st.error("❌ Nenhum documento encontrado na pasta ./docs")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = load_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTORDB_PATH)
    return db

# ✅ Carrega ou cria base vetorial
@st.cache_resource
def load_vectorstore():
    if os.path.exists(os.path.join(VECTORDB_PATH, "index.faiss")):
        embeddings = load_embeddings()
        return FAISS.load_local(
            VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True
        )
    else:
        return create_vectorstore()

# 🚀 Inicializa LLM e Base de Dados Vetorial
llm = load_llm()
vectorstore = load_vectorstore()

# 🔁 Chain de RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_type="similarity", k=4),
    return_source_documents=False
)

# 🧠 Interface de Chat
st.title("🧠 Chat com seus Documentos (RAG + GPT)")
st.markdown("📁 Lendo arquivos da pasta `./docs/`")

# 💬 Histórico de mensagens
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 📝 Entrada de pergunta
with st.form("chat-form", clear_on_submit=True):
    user_input = st.text_input("Digite sua pergunta:")
    submitted = st.form_submit_button("Enviar")

if submitted and user_input:
    resposta = qa_chain.run(user_input)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", resposta))

# 💬 Exibe histórico estilo chat
for role, msg in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(msg)

# 🧹 Botão para limpar chat
if st.button("🗑️ Limpar conversa"):
    st.session_state.chat_history = []

# 💾 Download da última resposta
if st.session_state.chat_history:
    for role, msg in reversed(st.session_state.chat_history):
        if role == "bot":
            st.download_button("📥 Baixar última resposta", msg, file_name="resposta.txt")
            break
