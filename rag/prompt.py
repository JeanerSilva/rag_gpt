from langchain.prompts import PromptTemplate

def get_custom_prompt():
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
Você é um assistente com acesso a trechos de documentos oficiais.

Seu trabalho é responder com base **exclusivamente no conteúdo abaixo**, sem adicionar informações externas.

🔹 **Trechos disponíveis:**
{context}

🔹 **Pergunta:**
{question}

💡 **Instruções para resposta**:
- Se a resposta estiver expressa literalmente nos trechos, repita-a com clareza.
- Seja direto e conciso. Não diga "não há informações" se a frase está nos trechos.
- Se os trechos não contêm a resposta, apenas diga: "Os documentos não fornecem essa informação."

📝 **Resposta**:
"""
    )
