import os
import json
import numpy as np
import faiss
import requests
import re
import time

def get_embeddings(texts):
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        raise ValueError("Missing HF_TOKEN environment variable.")

    if isinstance(texts, str):
        texts = [texts]

    if not texts or len(texts) == 0:
        raise ValueError("texts cannot be empty")
    
    print("Generating embeddings via Hugging Face Inference API...")
 
    url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                headers=headers,
                json={"inputs": texts},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                embeddings = np.array(data, dtype="float32")
                return embeddings
            elif response.status_code == 503:
                print(f"Model loading, waiting... (attempt {attempt + 1}/{max_retries})")
                time.sleep(5)
                continue
            else:
                raise Exception(f"HF API error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"Request timeout (attempt {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                raise Exception("Request timeout after multiple attempts")
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e} (attempt {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                raise Exception(f"Request failed after multiple attempts: {e}")
            time.sleep(2)
    
    raise Exception("Failed to get embeddings after all retries")

def load_or_build_index(index_path, meta_path):
    if os.path.exists(index_path) and os.path.exists(meta_path):
        try:
            index = faiss.read_index(index_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print("FAISS index and metadata loaded successfully.")
            return index, meta
        except Exception as e:
            print(f"Error loading index/metadata: {e}")
            return None, None
    print("No existing FAISS index found.")
    return None, None

def semantic_search(index, meta, queries, topk=10):
    if not queries:
        raise ValueError("queries cannot be empty")
    
    if isinstance(queries, str):
        queries = [queries]
    topk = min(topk, len(meta))
    
    q_emb = get_embeddings(queries)
    norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    q_emb = q_emb / norms
    
    D, I = index.search(q_emb, topk)
    results = []
    for qi in range(len(queries)):
        items = []
        for idx in I[qi]:
            if 0 <= idx < len(meta):
                items.append(meta[idx])
        results.append(items)
    return results

def llm_rerank(query, docs, topk=10):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY environment variable.")
    
    # Input validation
    if not docs or len(docs) == 0:
        return []
    
    # Ensure topk doesn't exceed available docs
    topk = min(topk, len(docs))
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"Rank these documents by how relevant they are to '{query}':\n\n"
    for i, d in enumerate(docs, 1):
        title = d.get('title', d.get('name', 'Untitled'))
        prompt += f"{i}. {title}\n"
    prompt += "\nReturn the top 10 document numbers as a JSON list."
    
    print("Calling Groq LLM for reranking...")
   
    max_retries = 2
    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                },
                timeout=30
            )
            
            if r.status_code != 200:
                print(f"Groq API error {r.status_code}: {r.text}")
                if attempt == max_retries - 1:
                    print("Reranking failed, returning original order")
                    return docs[:topk]
                time.sleep(2)
                continue
            
            data = r.json()
            if "choices" not in data:
                print("Groq rerank error:", data)
                return docs[:topk]
            
            text = data["choices"][0]["message"]["content"]
            numbers = re.findall(r"\d+", text)
            indices = [int(n) - 1 for n in numbers[:topk] if n.isdigit()]
            reranked = [docs[i] for i in indices if 0 <= i < len(docs)]

            if len(reranked) < topk:
                remaining = [d for d in docs if d not in reranked]
                reranked.extend(remaining[:topk - len(reranked)])
            
            return reranked[:topk]
            
        except requests.exceptions.Timeout:
            print(f"Groq timeout (attempt {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                return docs[:topk]
            time.sleep(2)
        except Exception as e:
            print(f"Groq error: {e}")
            return docs[:topk]
    
    return docs[:topk]
