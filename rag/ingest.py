from sentence_transformers import SentenceTransformer
import faiss, pickle

documents = [
    "Reset passwords using the self-service portal.",
    "VPN requires active internet and valid credentials.",
    "Email access may be locked after multiple failed attempts."
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
vectors = embedder.encode(documents)

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

faiss.write_index(index, "rag/it_support.faiss")
pickle.dump(documents, open("rag/docs.pkl", "wb"))
