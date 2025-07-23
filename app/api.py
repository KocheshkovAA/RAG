from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.loader import load_and_split_documents
from app.embedder import build_or_load_vectorstore
from app.llm import get_llm
from app.rag import build_rag_chain
from app.config import CHROMA_PERSIST_DIR

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# Загружаем один раз при старте
if CHROMA_PERSIST_DIR.exists() and any(CHROMA_PERSIST_DIR.iterdir()):
    vectorstore = build_or_load_vectorstore([])
else:
    chunks = load_and_split_documents()
    vectorstore = build_or_load_vectorstore(chunks)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = get_llm()
rag = build_rag_chain(llm, retriever)

@app.post("/ask")
def ask(request: QueryRequest):
    result = rag({"query": request.query})
    return {
        "result": result["result"],
        "sources": [doc.metadata.get("page", "?") for doc in result["source_documents"]]
    }
