import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('vector.index')

with open('chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

def retrieve(query, top_k=3):
    query_vec = model.encode([query])
    _, I = index.search(np.array(query_vec), top_k)
    return [chunks[i] for i in I[0]]
