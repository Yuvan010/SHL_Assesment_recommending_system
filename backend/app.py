# backend/app.py
from flask import Flask, request, jsonify, send_from_directory, send_file
import os
import csv
from backend.embed_utils import get_embeddings, semantic_search, load_or_build_index, llm_rerank

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"))
INDEX_PATH = os.path.join(BASE_DIR, "shl_faiss.index")
META_PATH = os.path.join(BASE_DIR, "shl_meta.json")

session_cache = []
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    faiss_index, meta_data = load_or_build_index(INDEX_PATH, META_PATH)
    print("FAISS index and metadata loaded successfully.")
else:
    faiss_index, meta_data = None, None
    print("No existing FAISS index found. Please run build_index.py first.")

@app.route("/")
def serve_frontend():
    """Serve the frontend index.html"""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    """Main recommendation route with full JSON schema"""
    data = request.get_json()
    queries = data.get("queries", [])
    results = []

    for q in queries:
        q = str(q).strip()
        if faiss_index:
            top_docs = semantic_search(faiss_index, meta_data, [q], topk=10)[0]
            reranked = llm_rerank(q, top_docs)
            formatted = []
            for doc in reranked[:10]:
                meta = next((m for m in meta_data if m.get("url") == doc["url"]), {})
                formatted.append({
                    "url": meta.get("url", ""),
                    "name": meta.get("name", ""),
                    "adaptive_support": meta.get("adaptive_support", "No"),
                    "description": meta.get("description", ""),
                    "duration": meta.get("duration", ""),
                    "remote_support": meta.get("remote_support", "Yes"),
                    "test_type": meta.get("test_type", ["Knowledge & Skills"])
                })

            results.extend(formatted)

            for item in formatted:
                session_cache.append({
                    "Query": q,
                    "Assessment_url": item.get("url", "")
                })

        else:
            fallback = {
                "url": f"https://www.shl.com/search?q={q.replace(' ', '+')}",
                "name": q,
                "adaptive_support": "No",
                "description": "Search fallback — no FAISS index available.",
                "duration": "",
                "remote_support": "Yes",
                "test_type": ["General"]
            }
            results.append(fallback)

            session_cache.append({
                "Query": q,
                "Assessment_url": fallback["url"]
            })

    return jsonify({"recommended_assessments": results})
@app.route("/download_csv", methods=["GET"])
def download_csv():
    if not session_cache:
        return jsonify({"error": "No recommendations found in this session."}), 400

    output_folder = os.path.join(BASE_DIR, "static", "output")
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, "session_results.csv")

    with open(file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["Query", "Assessment_url"])
        writer.writeheader()
        writer.writerows(session_cache)

    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    print(">> Starting SHL Recommender backend with frontend support...")
    app.run(host="0.0.0.0", port=5000, debug=True)
