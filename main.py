from src.engine import Engine
from src.model import Model

import json

with open("test/cats.json", "r") as file:
    data = json.load(file)

model = Model("all-MiniLM-L6-v2", local_folder_path='./local')
engine = Engine(model.encode)
engine.add(data["documents"])

def search(query):
    results = engine.search(query, top_k=10)

    print(f"Search Query: '{query}'")
    print("-" * 50)
    for result in results:
        print(f"Score: {result['score']:.4f} | {result['query']}")
    print("=" * 50)

for query in data["queries"]:
    search(query)