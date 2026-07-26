import json
import hashlib
import uuid
import httpx
from qdrant_client import QdrantClient
from app.core.embedder import TEIEmbeddings, BM25SparseEmbeddings
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, PointStruct, OptimizersConfigDiff
from app.core.config import settings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from qdrant_client.http.exceptions import ResponseHandlingException

from itertools import islice

# Инициализация клиента с увеличенным таймаутом (60 секунд)
client = QdrantClient(url=settings.QDRANT_URL, timeout=60)
embeddings_dense = TEIEmbeddings()
embeddings_sparse = BM25SparseEmbeddings()

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ResponseHandlingException, Exception))
)
def safe_upsert(points):
    """Обертка для безопасной вставки в Qdrant с ретраями"""
    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(httpx.HTTPStatusError),
)
def safe_embed_dense(texts):
    """TEI отдаёт 429 при перегруженной очереди (особенно на CPU-фолбэке) — ретраим с backoff"""
    return embeddings_dense.embed_documents(texts)

def generate_deterministic_uuid(text: str):
    hash_obj = hashlib.md5(text.encode())
    return str(uuid.UUID(hash_obj.hexdigest()))

def process_batch(batch):
    texts = [item["text"] for item in batch]

    dense_vectors = safe_embed_dense(texts)
    sparse_vectors = embeddings_sparse.embed_documents(texts)
    
    points = []
    for idx, item in enumerate(batch):
        content = item.pop("text")
        if "meta" in item:
            item.update(item.pop("meta")) if isinstance(item["meta"], dict) else None
        
        points.append(PointStruct(
            id=generate_deterministic_uuid(content),
            vector={
                "text-dense": dense_vectors[idx],
                "text-sparse": sparse_vectors[idx]
            },
            payload={
                "content": content,
                "metadata": item,
                "source": item.get("source", "warhammer_wiki")
            }
        ))
    
    safe_upsert(points)

def run_ingestion():
    print(f"🚀 Starting ingestion to collection: {settings.QDRANT_COLLECTION}")
    
    if client.collection_exists(settings.QDRANT_COLLECTION):
        print(f"♻️ Collection {settings.QDRANT_COLLECTION} exists. Re-creating...")
        client.delete_collection(settings.QDRANT_COLLECTION)
    
    # Создаем коллекцию
    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config={
            "text-dense": VectorParams(size=settings.VECTOR_SIZE, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "text-sparse": SparseVectorParams(index=SparseIndexParams(on_disk=True))
        }
    )

    # Оптимизация: временно отключаем индексацию для быстрой заливки тяжелых векторов
    client.update_collection(
        collection_name=settings.QDRANT_COLLECTION,
        optimizer_config=OptimizersConfigDiff(indexing_threshold=0)
    )

    try:
        with open(settings.DATA_PATH, "r", encoding="utf-8") as f:
            batch = []
            batch_size = 10 
            
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    batch.append(data)
                except json.JSONDecodeError:
                    continue
                
                if len(batch) >= batch_size:
                    process_batch(batch)
                    batch = []
            
            if batch:
                process_batch(batch)
        
        # Возвращаем индексацию в стандартный режим после заливки
        print("🏗 Building HNSW index...")
        client.update_collection(
            collection_name=settings.QDRANT_COLLECTION,
            optimizer_config=OptimizersConfigDiff(indexing_threshold=200)
        )
        
        print("🏁 Ingestion complete!")
    except FileNotFoundError:
        print(f"❌ Error: File not found at {settings.DATA_PATH}.")

if __name__ == "__main__":
    run_ingestion()