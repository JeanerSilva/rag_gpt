from langchain.chains import RetrievalQA
from rag.prompt import get_custom_prompt

def build_qa_chain(llm, vectorstore):
    if not vectorstore:
        return None
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", k=6),
        return_source_documents=True,
        chain_type_kwargs={"prompt": get_custom_prompt()}
    )
