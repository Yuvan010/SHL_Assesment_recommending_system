# backend/build_index.py
import os, json, numpy as np, faiss, pandas as pd, time, re, requests
from bs4 import BeautifulSoup
from embed_utils import get_embeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../Gen_AI Dataset.xlsx")
INDEX_PATH = os.path.join(BASE_DIR, "shl_faiss.index")
META_PATH = os.path.join(BASE_DIR, "shl_meta.json")

def scrape_shl_page(url):
    """Extract metadata such as duration, description, and test type from SHL assessment URL."""
    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return {}

        soup = BeautifulSoup(res.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        duration_match = re.search(r"(\d{1,2})\s*(minutes|min)", text, re.IGNORECASE)
        duration = int(duration_match.group(1)) if duration_match else None
        desc_tag = soup.find("meta", {"name": "description"})
        description = desc_tag["content"].strip() if desc_tag and desc_tag.get("content") else ""
        test_type = []
        if re.search("knowledge", text, re.IGNORECASE):
            test_type.append("Knowledge & Skills")
        if re.search("personality", text, re.IGNORECASE):
            test_type.append("Personality & Behaviour")
        if re.search("reasoning|logic", text, re.IGNORECASE):
            test_type.append("Cognitive Ability")
        if not test_type:
            test_type.append("General")

        return {
            "duration": duration,
            "description": description,
            "test_type": test_type
        }

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return {}
df = pd.read_excel(DATA_PATH)
print("Columns:", df.columns.tolist())
text_col = None
for c in df.columns:
    if "query" in c.lower() or "details" in c.lower() or "description" in c.lower():
        text_col = c
        break
if not text_col:
    text_col = df.columns[0]

texts = df[text_col].astype(str).tolist()
meta = []
for i, row in df.iterrows():
    url = str(row.get("Assessment_url", "")).strip()
    if not url or not url.startswith("http"):
        continue

    info = scrape_shl_page(url)
    meta.append({
        "url": url,
        "name": str(row.get("Query", "")),
        "description": info.get("description", ""),
        "adaptive_support": "No",
        "remote_support": "Yes",
        "duration": info.get("duration", ""),
        "test_type": info.get("test_type", ["General"])
    })

    print(f"Processed {i+1}/{len(df)} — {url} ({info.get('duration', 'N/A')} mins)")
print("Generating embeddings...")
embeddings = get_embeddings(texts)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)
json.dump(meta, open(META_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print(f"\n FAISS index saved to: {INDEX_PATH}")
print(f" Metadata saved to: {META_PATH}")
print(f"Entries: {len(meta)}")
