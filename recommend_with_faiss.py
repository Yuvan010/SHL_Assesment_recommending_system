import argparse, json, numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--queries", nargs="+", required=True)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    index = faiss.read_index(args.index)
    metas = json.load(open(args.meta, 'r', encoding='utf-8'))
    model = SentenceTransformer(args.model)
    q_emb = model.encode(args.queries, convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, args.topk)
    for qi, q in enumerate(args.queries):
        print("Query:", q)
        for rank, idx in enumerate(I[qi]):
            if idx < 0 or idx >= len(metas): continue
            m = metas[idx]
            print(f" {rank+1}. {m.get('title')} - {m.get('url')} (score {D[qi][rank]:.4f})")
        print()

if __name__ == '__main__':
    main()