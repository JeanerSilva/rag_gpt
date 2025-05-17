import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Diretórios
DOCS_PATH = "./docs"
VECTORDB_PATH = "./vectordb"
INDEXED_LIST_PATH = "indexed_files.json"

os.makedirs(DOCS_PATH, exist_ok=True)
os.makedirs(VECTORDB_PATH, exist_ok=True)

# API
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

def load_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=openai_key,
    )
