# settings.py

# Número de chunks retornados pelo FAISS retriever
# O parâmetro k define quantos trechos mais relevantes o sistema deve recuperar da base de documentos indexados para responder 
# a uma pergunta. Esses trechos são escolhidos com base em similaridade com a consulta do usuário. Um valor mais alto de k aumenta a
# chance de trazer informações úteis, mas também pode incluir conteúdo irrelevante. Um valor mais baixo reduz ruído, mas pode omitir 
# partes importantes. Recomenda-se ajustar k com base na complexidade dos documentos e das perguntas esperadas. 
# Valores típicos variam entre 4 e 10.
RETRIEVER_TOP_K = 10


# Parâmetros do text splitter
# Antes de indexar, os documentos são divididos em partes menores chamadas “chunks”. 
# O chunk_size define o tamanho de cada bloco (em caracteres), e o chunk_overlap define o quanto cada bloco se sobrepõe ao anterior. 
# Essa sobreposição é importante para manter o contexto entre blocos vizinhos. Um chunk_size menor pode facilitar a recuperação de 
# informações pontuais, enquanto valores maiores mantêm mais contexto. Para documentos com conteúdo denso ou técnico, 
# recomenda-se começar com chunk_size = 500 e chunk_overlap = 100, ajustando conforme necessário.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
