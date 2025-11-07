import requests, time, argparse, json, re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import sys

BASE = "https://www.shl.com"
CATALOG = "https://www.shl.com/products/product-catalog/"

HEADERS = {"User-Agent": "SHL-Catalog-Crawler/1.0 (+https://example.com)"} 
def allowed_by_robots(url):

    rp = urlparse(url)
    robots_url = f"{rp.scheme}://{rp.netloc}/robots.txt"
    try:
        r = requests.get(robots_url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return True
        txt = r.text
        if "Disallow:" in txt:
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            for i, line in enumerate(lines):
                if line.lower().startswith("user-agent:"):
                    ua = line.split(":",1)[1].strip()
            if "Disallow: /products/" in txt:
                return False
        return True
    except Exception as e:
        print("Robots check failed, proceeding cautiously:", e)
        return True

def parse_product_page(html, url):
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find(['h1','h2'], string=True)
    title = title_tag.get_text(strip=True) if title_tag else None
    desc = None
    desc_h = soup.find(lambda tag: tag.name in ['h2','h3','h4'] and 'Description' in tag.get_text())
    if desc_h:
        p = desc_h.find_next_sibling(['p','div'])
        if p:
            desc = p.get_text(" ", strip=True)
    text = soup.get_text(" ", strip=True)
    test_type = None
    m = re.search(r"Test Type:\\s*([A-Z\\s]+)", text)
    if m:
        test_type = m.group(1).strip()
    job_levels = None
    m2 = re.search(r"Job levels\\s*(.*?)Test Type", text, re.S)
    if m2:
        job_levels = m2.group(1).strip()
    return {
        "url": url,
        "title": title,
        "description": desc,
        "test_type": test_type,
        "job_levels": job_levels,
    }

def fetch_catalog_page(page_num=1):
    params = {}
    if page_num > 1:
        params['page'] = page_num
    r = requests.get(CATALOG, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.text

def extract_product_links(catalog_html):
    soup = BeautifulSoup(catalog_html, "html.parser")
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if '/products/product-catalog/view/' in href:
            full = urljoin(BASE, href)
            if full not in links:
                links.append(full)
    return links

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=32, help="Maximum catalog pages to fetch (safety cap)")
    parser.add_argument("--out", type=str, default="products.json", help="Output JSON file")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    args = parser.parse_args()

    if not allowed_by_robots(CATALOG):
        print("Crawling disallowed by robots.txt. Exiting.")
        sys.exit(1)

    all_product_urls = []
    for p in range(1, args.pages+1):
        print("Fetching catalog page", p)
        try:
            html = fetch_catalog_page(page_num=p)
        except Exception as e:
            print("Failed to get catalog page", p, e)
            break
        links = extract_product_links(html)
        print("Found", len(links), "product links on page", p)
        for l in links:
            if l not in all_product_urls:
                all_product_urls.append(l)
        time.sleep(args.delay)

    print("Total unique product URLs discovered:", len(all_product_urls))
    products = []
    for i, url in enumerate(all_product_urls):
        print(f"[{i+1}/{len(all_product_urls)}] Fetching product {url}")
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            prod = parse_product_page(r.text, url)
            products.append(prod)
        except Exception as e:
            print("Failed to fetch product page", url, e)
        time.sleep(args.delay)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    print("Wrote", args.out)

if __name__ == '__main__':
    main()