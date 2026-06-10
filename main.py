import os
from model import Model
import pickle

filename = 'model-cat.pkl'

documents = [
    "cat",
    "furry cat",
    "cataclysm",
    "a small domesticated feline animal",
    "a kitten playing with yarn",
    "a catastrophic natural disaster destroying cities",
    "cats are independent pets often kept indoors",
    "big cats like lions and tigers are wild animals",
    "cathedral architecture in medieval europe",
    "the CAT exam is a competitive test in India"
]

queries = [
    "cat",
    "kitty",
    "feline animal",
    "furry animal",
    "cat animal pet",
    "catastrophe",
    "natural disaster",
    "cat exam India",
    "big wild cats",
    "cathedral building",
    "cataclysm meaning",
    "cats as pets",
    "lion tiger animals",
    "domestic kitten",
    "exam for management admission",
    "very hard test for south asia"
]

if os.path.exists(filename):
    print("found model file")

    with open(filename, "rb") as f:
        mod = pickle.load(f)
else:
    print('found no model file')
    
    mod = Model(model_name='all-MiniLM-L6-v2')

    print("Encoding documents into vectors...")
    mod += documents

def search(query):
    results = mod.search(query, top_k=10)

    print(f"Search Query: '{query}'")
    print("-" * 50)
    for result in results:
        print(f"Score: {result['score']:.4f} | {result['query']}")
    print("=" * 50)

for query in queries:
    search(query)

with open(filename, "wb") as f:
    pickle.dump(mod, f)