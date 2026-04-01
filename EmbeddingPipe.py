
import torch.nn.functional as F
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