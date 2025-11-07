import os
import json
import numpy as np
import faiss
import requests

def get_embeddings(texts):
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        raise ValueError("Missing HF_API_KEY environment variable.")
    
    print("Generating embeddings via Hugging Face Inference API...")
    response = requests.post(
        "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"inputs": texts}
    )

    if response.status_code != 200:
        raise Exception(f"HF API error {response.status_code}: {response.text}")

    data = response.json()
    if not isinstance(data, list):
        raise Exception(f"Unexpected response format: {data}")

    embeddings = np.array(data, dtype="float32")
    return embeddings


def load_or_build_index(index_path, meta_path):
    if os.path.exists(index_path) and os.path.exists(meta_path):
        index = faiss.read_index(index_path)
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        print("FAISS index and metadata loaded successfully.")
        return index, meta
    print("No existing FAISS index found.")
    return None, None


def semantic_search(index, meta, queries, topk=10):
    q_emb = get_embeddings(queries)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    D, I = index.search(q_emb, topk)
    results = []
    for qi in range(len(queries)):
        items = []
        for idx in I[qi]:
            items.append(meta[idx])
        results.append(items)
    return results


def llm_rerank(query, docs, topk=10):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY environment variable.")

    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = f"Rank these documents by how relevant they are to '{query}':\n\n"
    for i, d in enumerate(docs, 1):
        prompt += f"{i}. {d.get('title', '')}\n"
    prompt += "\nReturn the top 10 document numbers as a JSON list."

    print("Calling Groq LLM for reranking...")
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
