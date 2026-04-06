import faiss
from helper import setup_device, make_faiss_index


class Faiss128Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Faiss128Singleton, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Called only once during the first initialization."""
        print("Initializing 128d FAISS index (Singleton)...")
        self.device = setup_device()
        self.dim = 128
        self.index = make_faiss_index(self.dim, self.device)
        self.is_populated = False

    def populate(self, base_embeddings):
        """Populate the index with base embeddings if not already populated."""
        if not self.is_populated:
            print(f"Adding {len(base_embeddings)} vectors to the index...")
            self.index.add(base_embeddings)
            self.is_populated = True
        else:
            print("Index is already populated. Skipping...")

    def search(self, query_embeddings, k=5):
        """Convenience search function."""
        if not self.is_populated:
            raise ValueError("The index is empty! Call populate() first.")
        return self.index.search(query_embeddings, k)

    def get_index(self):
        """Return the raw FAISS index object."""
        return self.index