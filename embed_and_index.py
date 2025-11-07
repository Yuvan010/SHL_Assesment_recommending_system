import argparse, json, os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def load_products(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def make_corpus(products):
    corpus = []
    metas = []
    for p in products:
        text = " ".join(filter(None, [p.get('title',''), p.get('description','') or '', p.get('test_type') or '']))
        corpus.append(text)
        metas.append({'url': p.get('url'), 'title': p.get('title')})
    return corpus, metas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", required=True)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--out", default="index.faiss")
    parser.add_argument("--meta", default="meta.json")
    args = parser.parse_args()

    products = load_products(args.products)
    corpus, metas = make_corpus(products)
    model = SentenceTransformer(args.model)
    embeddings = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine via inner product on normalized vectors
    index.add(embeddings)
    faiss.write_index(index, args.out)
    with open(args.meta, 'w', encoding='utf-8') as f:
        json.dump(metas, f, indent=2, ensure_ascii=False)
    print("Wrote index and meta files:", args.out, args.meta)

if __name__ == '__main__':
    main()