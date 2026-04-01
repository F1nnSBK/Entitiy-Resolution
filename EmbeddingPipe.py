import torch
import numpy as np
from transformers import AutoModel


class JinaEmbeddingPipeline:
    def __init__(self, model_name="jinaai/jina-embeddings-v4"):
        print(f"Lade Modell '{model_name}'...")
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.float16,
        ).to(self.device)
        self.model.eval()
        print(f"Modell geladen auf {self.device}.")

    def encode(self, texts: list[str], batch_size: int = 8, task: str = "text-matching") -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.model.encode_text(texts=batch, task=task)
            if isinstance(embeddings, torch.Tensor):
                all_embeddings.append(embeddings.cpu().float().numpy())
            else:
                # Liste von Tensors abfangen
                arr = np.array([
                    e.cpu().float().numpy() if isinstance(e, torch.Tensor) else e
                    for e in embeddings
                ])
                all_embeddings.append(arr)
        return np.vstack(all_embeddings)


pipeline = JinaEmbeddingPipeline()


def truncate_and_normalize(embeddings: np.ndarray, dim: int) -> np.ndarray:
    truncated = embeddings[:, :dim]
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    return truncated / np.maximum(norms, 1e-9)