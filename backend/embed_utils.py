import os
import json
import numpy as np
import faiss
import requests

# --------------------------
# Embedding Configuration
# --------------------------
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN")  # Set this in Render environment variables
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# --------------------------
# Embedding Function (Hugging Face API)
# --------------------------
def get_embeddings(texts):
    """
    Generate embeddings for a list of texts using Hugging Face's Inference API.
    Caches embeddings locally to avoid repeated API calls.
    """

    if not HF_TOKEN:
        raise ValueError("❌ Missing HF_TOKEN environment variable. Please set it in Render.")

    if isinstance(texts, str):
        texts = [texts]

    print(f"🔗 Generating embeddings using Hugging Face model: {HF_MODEL}")

    all_vecs = []

    for text in texts:
        # simple caching mechanism
        cache_dir = "hf_embed_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{abs(hash(text))}.npy")

        if os.path.exists(cache_file):
            vec = np.load(cache_file)
        else:
            response = requests.post(
                HF_API_URL,
                headers=HEADERS,
                json={"inputs": text},
                timeout=60,
            )

            if response.status_code != 200:
                print("❌ Hugging Face API Error:", response.text)
                raise Exception(f"HF API error {response.status_code}: {response.text}")

            vec = np.array(response.json(), dtype="float32")
            np.save(cache_file, vec)

        all_vecs.append(vec)

    return np.vstack(all_vecs)

# --------------------------
# FAISS Index Functions
# --------------------------
def load_or_build_index(index_path, meta_path):
    if os.path.exists(index_path) and os.path.exists(meta_path):
        index = faiss.read_index(index_path)
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        print("✅ FAISS index and metadata loaded successfully.")
        return index, meta
    print("⚠️ No existing FAISS index found.")
    return None, None

# --------------------------
# Semantic Search
# --------------------------
def semantic_search(index, meta, queries, topk=10):
    q_emb = get_embeddings(queries)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    D, I = index.search(q_emb, topk)

    results = []
    for qi in range(len(queries)):
        items = []
        for idx in I[qi]:
            if idx < len(meta):
                items.append(meta[idx])
        results.append(items)
    return results

# --------------------------
# LLM Reranking (Groq API)
# --------------------------
def llm_rerank(query, docs, topk=10):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY environment variable.")

    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = f"Rank these documents by how relevant they are to '{query}':\n\n"
    for i, d in enumerate(docs, 1):
        prompt += f"{i}. {d.get('title', d.get('name', ''))}\n"
    prompt += "\nReturn the top 10 document numbers as a JSON list."

    print("⚙️ Calling Groq LLM for reranking...")

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
    )

    data = r.json()
    if "choices" not in data:
        print("Groq rerank error:", data)
        return docs[:topk]

    text = data["choices"][0]["message"]["content"]
    import re
    numbers = re.findall(r"\d+", text)
    indices = [int(n) - 1 for n in numbers[:topk] if n.isdigit()]

    return [docs[i] for i in indices if 0 <= i < len(docs)]
