import copy
import numpy as np

class Engine:
    def __init__(self, encodeFunction):
        self.queries = []
        self.encodeFunction = encodeFunction
        
        dim = encodeFunction([""])[0].shape[0]
        self.vectors = np.empty((0, dim), dtype=np.float32)

    def add(self, document):
        if isinstance(document, str):
            document = [document]
        elif not isinstance(document, list):
            raise TypeError("Only str or list allowed")

        vecs = self.encodeFunction(document)
        self.queries.extend(document)
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
        if not self.queries:
            return []

        q = self.encodeFunction([query])[0]
        scores = self.vectors @ q

        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            {"query": self.queries[i], "score": float(scores[i])}
            for i in top_idx
        ]
    
    def __getstate__(self):
        state = self.__dict__.copy()
        if "encodeFunction" in state:
            del state["encodeFunction"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.encodeFunction = None