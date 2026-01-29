import faiss, pickle
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("rag/it_support.faiss")
docs = pickle.load(open("rag/docs.pkl", "rb"))

def retrieve_it_context(query, k=2):
    q = embedder.encode([query])
    _, idx = index.search(q, k)
    return "\n".join([docs[i] for i in idx[0]])
