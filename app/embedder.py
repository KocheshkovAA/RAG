import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def build_or_load_vectorstore(chunks):
    if CHROMA_PERSIST_DIR.exists() and any(CHROMA_PERSIST_DIR.iterdir()):
        return Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=embedding_model
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=str(CHROMA_PERSIST_DIR)
        )
        vectorstore.persist()
        return vectorstore
