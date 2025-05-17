from langchain_huggingface import HuggingFaceEmbeddings

def load_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
