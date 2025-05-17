import os
import json
from config import DOCS_PATH, INDEXED_LIST_PATH

def save_uploaded_files(uploaded_files):
    for file in uploaded_files:
        with open(os.path.join(DOCS_PATH, file.name), "wb") as f:
            f.write(file.getvalue())

def save_indexed_files(file_list):
    with open(INDEXED_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(file_list, f, ensure_ascii=False, indent=2)

def load_indexed_files():
    if os.path.exists(INDEXED_LIST_PATH):
        with open(INDEXED_LIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []
