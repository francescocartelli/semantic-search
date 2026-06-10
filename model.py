import copy
import numpy as np
from sentence_transformers import SentenceTransformer

class Model:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.queries = []
        self.vectors = np.zeros((0, 384), dtype=np.float32)  # updated on first encode

    def _encode(self, texts):
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.array(vecs, dtype=np.float32)

    def add(self, document):
        if isinstance(document, str):
            document = [document]
        elif not isinstance(document, list):
            raise TypeError("Only str or list allowed")

        vecs = self._encode(document)

        self.queries.extend(document)

        if self.vectors.shape[0] == 0:
            self.vectors = vecs
        else:
            self.vectors = np.vstack([self.vectors, vecs])

    def remove(self, document):
        if isinstance(document, str):
            document = [document]
        elif not isinstance(document, list):
            raise TypeError("Only str or list allowed")

        remove_set = set(document)

        keep_idx = [i for i, item in enumerate(self.queries) if item not in remove_set]

        self.queries = [self.queries[i] for i in keep_idx]
        self.vectors = self.vectors[keep_idx]

    def __add__(self, document):
        new = copy.copy(self)
        new.queries = self.queries.copy()
        new.vectors = self.vectors.copy()
        new.add(document)
        return new

    def __sub__(self, document):
        new = copy.copy(self)
        new.queries = self.queries.copy()
        new.vectors = self.vectors.copy()
        new.remove(document)
        return new

    def __iadd__(self, document):
        self.add(document)
        return self

    def __isub__(self, document):
        self.remove(document)
        return self

    def search(self, query, top_k=5):
        if len(self.queries) == 0:
            return []

        q = self._encode([query])[0]

        scores = self.vectors @ q  # cosine because normalized

        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            {"query": self.queries[i], "score": float(scores[i])}
            for i in top_idx
        ]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("model", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = SentenceTransformer(self.model_name)