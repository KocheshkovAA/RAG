import logging
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class MLMEmbeddings:
    """
    Обёртка для MLM модели, чтобы можно было получать эмбеддинги для Chroma.
    Делает mean pooling по токенам с учётом attention mask.
    """
    def __init__(self, model_path: str, device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embs = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (last_hidden * attention_mask).sum(1) / attention_mask.sum(1)
                all_embs.append(pooled.squeeze(0).cpu().tolist())
        return all_embs

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
    
    def embed_documents_batch(self, texts: list[str], batch_size=32):
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (last_hidden * attention_mask).sum(1) / attention_mask.sum(1)
                all_embs.extend(pooled.cpu().tolist())
        return all_embs

