import os
import numpy as np
from sentence_transformers import SentenceTransformer

class Model:
    def __init__(self, model_name, local_folder_path=None):
        self.model_name = model_name
        self.model = None

        if local_folder_path is None:
            self.path = None
            self.load(self.model_name)
            return
        
        self.path = os.path.join(local_folder_path, model_name)
        os.makedirs(local_folder_path, exist_ok=True)
        
        if os.path.exists(self.path):
            self.load(self.path)
            return
            
        self.load(self.model_name)
        self.save(self.path)

    def encode(self, texts):
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.array(vecs, dtype=np.float32)

    def load(self, path):
        self.model = SentenceTransformer(path)

    def __getattr__(self, name):
        return getattr(self.model, name)