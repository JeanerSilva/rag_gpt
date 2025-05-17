FROM python:3.10-slim

# Evita criação de bytecode e buffers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    python3-dev \
    libgl1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Instala Tesseract para Unstructured se necessário (OCR para PDF digitalizado)
RUN apt-get install -y tesseract-ocr

# Define diretório da aplicação
WORKDIR /app

# Copia os arquivos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do app
COPY . .

# Expõe a porta do Streamlit
EXPOSE 8501

# Comando de execução
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
