from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return splitter.split_text(text)

def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, show_progress_bar=True)
    return model, embeddings

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def save_data(index, chunks):
    faiss.write_index(index, 'vector.index')
    with open('chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)

if __name__ == "__main__":
    text = load_document('document.txt')
    chunks = chunk_text(text)
    model, embeddings = embed_chunks(chunks)
    index = create_faiss_index(embeddings)
    save_data(index, chunks)
    print("Document processed and indexed.")
