import torch
import numpy as np
from transformers import AutoModel


class JinaEmbeddingPipeline:
    def __init__(self, model_name="jinaai/jina-embeddings-v4"):
        print(f"Lade Modell '{model_name}'...")
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()
        print(f"Modell geladen auf {self.device}.")

    @torch.no_grad()
    def encode(self, texts, batch_size=32) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.model.encode(
                batch_texts, 
                task="text_matching",
                return_tensors="pt"
            )
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)
    
EmbeddingPipe = JinaEmbeddingPipeline()

def truncate_and_normalize(embeddings, dim) -> np.ndarray:
    truncated = embeddings[:, :dim]
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    return truncated / norms