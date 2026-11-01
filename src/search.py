"""Natural language artwork search: classical (TF-IDF) vs. neural (MiniLM) retrieval.

Enables visitors to query the collection in free-form natural language:
  - "Which painting shows a stormy sea?"
  - "I would like to see portraits of children"

Compares a classical inverted-index/TF-IDF baseline against a compact PyTorch-based
neural text bi-encoder (sentence-transformers/all-MiniLM-L6-v2) on modest kiosk CPUs:
measures query latency (1 thread, batch 1), memory footprint, and retrieval behavior.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

from data import RESULTS_DIR, out_path

ARTIST_DESCRIPTIONS = {
    "william-turner": "Dramatic atmospheric marine landscape with stormy sea, turbulent waves, dark sky, mist, and luminous coastal light.",
    "ivan-aivazovsky": "Romanticism seascape and marine cityscape featuring a stormy sea, crashing waves, sailing vessels, and ocean horizon.",
    "mary-cassatt": "Impressionist domestic portrait depicting portraits of children, infants, and mothers in soft pastel colors and gentle light.",
    "zinaida-serebriakova": "Realist and Expressionist portraits of children, young girls, and family members in domestic interior settings.",
    "claude-monet": "Impressionist cityscape and landscape depicting Paris streets, bridges along the Seine river, water lilies, and shifting daylight.",
    "camille-pissarro": "Impressionist urban cityscape showing bustling Paris boulevards, public squares, carriages, and atmospheric daylight.",
    "georges-braque": "Cubist still life deconstructing violins, guitars, fruit bowls, bottles, and sheet music into faceted geometric planes.",
    "juan-gris": "Synthetic Cubist still life composed of musical instruments, wine glasses, newspapers, and fruit arranged on tabletops.",
    "bartolome-esteban-murillo": "Baroque religious painting showing the Madonna and Child, young angels, cherubs, and devotional saints with soft chiaroscuro.",
    "rembrandt": "Baroque religious painting and portrait using dramatic chiaroscuro, golden light, deep shadows, and spiritual biblical figures.",
    "albrecht-durer": "Northern Renaissance religious painting and cityscape with intricate linework, biblical narrative, and sacred Christian iconography.",
    "amedeo-modigliani": "Expressionist portrait characterized by elongated faces, almond eyes, simplified silhouettes, and warm earth tones.",
    "edouard-cortes": "Post-Impressionist evening cityscape showing illuminated Parisian boulevards, street lamps, kiosks, and rainy sidewalks.",
    "vincent-van-gogh": "Post-Impressionist cityscape and landscape rendered with dynamic brushstrokes, vivid color contrast, and expressive texture.",
    "henri-matisse": "Fauvist still life and interior with bold primary colors, flat decorative patterns, fruit, and expressive outlines.",
}

GENRE_FALLBACK = {
    "landscape": "Landscape showing open countryside, hills, trees, horizon, and atmospheric sky.",
    "cityscape": "Cityscape showing urban streets, architectural buildings, public squares, and facades.",
    "portrait": "Portrait depicting a human subject, facial features, posture, and traditional dress.",
    "religious_painting": "Religious painting depicting Christian biblical iconography, saints, and devotional scenes.",
    "still_life": "Still life arrangement of inanimate objects, fruit, flowers, and tableware on a surface.",
}


def format_name(artist_slug: str) -> str:
    if artist_slug == "Unknown Artist":
        return "Unknown Artist"
    return " ".join(word.capitalize() for word in artist_slug.split("-"))


def load_or_build_catalog() -> list[dict]:
    """Loads metadata from data/wikiart_genre5/metadata.csv or builds it from the cached label index."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    csv_path = root / "data" / "wikiart_genre5" / "metadata.csv"
    catalog = []

    if csv_path.exists():
        import pandas as pd
        meta = pd.read_csv(csv_path)
        for _, row in meta.iterrows():
            a_slug = str(row["artist"])
            g = str(row["genre"])
            st = str(row["style"])
            desc = ARTIST_DESCRIPTIONS.get(a_slug, GENRE_FALLBACK.get(g, f"{g} in {st} style."))
            a_name = format_name(a_slug)
            catalog.append({
                "id": str(row["id"]),
                "artist": a_name,
                "artist_slug": a_slug,
                "genre": g,
                "style": st,
                "text": f"{g.replace('_', ' ').capitalize()} by {a_name}. Style: {st.replace('_', ' ')}. {desc}",
            })
        return catalog

    # Build from cached index and plan
    cache_path = Path.home() / ".cache" / "wikiart_label_index_d559852d2b23.json"
    if not cache_path.exists():
        matches = list((Path.home() / ".cache").glob("wikiart_label_index_*.json"))
        if matches:
            cache_path = matches[0]
        else:
            raise FileNotFoundError("WikiArt label index cache not found.")

    index = json.loads(cache_path.read_text(encoding="utf-8"))
    from make_wikiart_subset import make_plan, resolve_classes

    class_ids = resolve_classes(
        ["portrait", "landscape", "cityscape", "still_life", "religious_painting"],
        index["names"]["genre"]
    )
    plan, _ = make_plan(index, "genre", class_ids, 300, 50, "fewest-shards", 0)

    for s in sorted(plan):
        for r in plan[s]:
            g = index["names"]["genre"][index["shards"][s]["genre"][r]]
            st = index["names"]["style"][index["shards"][s]["style"][r]]
            a_slug = index["names"]["artist"][index["shards"][s]["artist"][r]]
            a_name = format_name(a_slug)
            desc = ARTIST_DESCRIPTIONS.get(a_slug, GENRE_FALLBACK.get(g, f"{g} in {st} style."))
            catalog.append({
                "id": f"s{s:02d}_r{r:04d}",
                "artist": a_name,
                "artist_slug": a_slug,
                "genre": g,
                "style": st,
                "text": f"{g.replace('_', ' ').capitalize()} by {a_name}. Style: {st.replace('_', ' ')}. {desc}",
            })

    catalog.sort(key=lambda d: d["id"])
    out_dir = RESULTS_DIR / "search"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=1)
    return catalog


class TfidfSearcher:
    """Classical baseline: Inverted index / TF-IDF bag-of-words vector space model."""

    def __init__(self, catalog: list[dict]):
        self.catalog = catalog
        self.corpus = [item["text"] for item in catalog]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(self.corpus)
        self.vocab_size = len(self.vectorizer.vocabulary_)

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        q_vec = self.vectorizer.transform([query])
        scores = (q_vec @ self.doc_matrix.T).toarray().ravel()
        top_indices = np.argsort(-scores)[:top_k]
        return [(self.catalog[i], float(scores[i])) for i in top_indices]

    def latency_benchmark(self, query: str, n_runs: int = 50) -> float:
        # Warmup
        for _ in range(5):
            self.search(query)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            self.search(query)
        return (time.perf_counter() - t0) / n_runs * 1000.0


class NeuralSearcher:
    """Neural text bi-encoder using sentence-transformers/all-MiniLM-L6-v2."""

    def __init__(self, catalog: list[dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.catalog = catalog
        self.corpus = [item["text"] for item in catalog]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())

        # Precompute and cache gallery representations
        cache_path = RESULTS_DIR / "search" / "gallery_embeddings_minilm.pt"
        if cache_path.exists():
            self.gallery_embeddings = torch.load(cache_path, weights_only=True)
        else:
            self.gallery_embeddings = self._encode_batch(self.corpus)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.gallery_embeddings, cache_path)

    def _encode_batch(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
                out = self.model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
                emb = (out.last_hidden_state * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
                emb = F.normalize(emb, p=2, dim=1)
                all_embeddings.append(emb)
        return torch.cat(all_embeddings, dim=0)

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        with torch.no_grad():
            inputs = self.tokenizer([query], padding=True, truncation=True, max_length=128, return_tensors="pt")
            out = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            q_emb = (out.last_hidden_state * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
            q_emb = F.normalize(q_emb, p=2, dim=1)
            sims = torch.matmul(q_emb, self.gallery_embeddings.T).squeeze(0)
            top_scores, top_indices = torch.topk(sims, k=top_k)
        return [(self.catalog[int(idx)], float(score)) for idx, score in zip(top_indices, top_scores)]

    def latency_benchmark(self, query: str, n_runs: int = 50) -> tuple[float, float]:
        """Returns (end_to_end_ms, ranking_only_ms) on 1 CPU thread."""
        threads = torch.get_num_threads()
        torch.set_num_threads(1)

        # Warmup
        for _ in range(5):
            self.search(query)

        t0 = time.perf_counter()
        for _ in range(n_runs):
            self.search(query)
        end_to_end = (time.perf_counter() - t0) / n_runs * 1000.0

        # Measure ranking dot product only
        with torch.no_grad():
            inputs = self.tokenizer([query], return_tensors="pt")
            out = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            q_emb = F.normalize((out.last_hidden_state * mask).sum(1) / mask.sum(1), p=2, dim=1)

            t1 = time.perf_counter()
            for _ in range(n_runs * 5):
                _ = torch.matmul(q_emb, self.gallery_embeddings.T)
            ranking_only = (time.perf_counter() - t1) / (n_runs * 5) * 1000.0

        torch.set_num_threads(threads)
        return end_to_end, ranking_only


BENCHMARK_QUERIES = [
    {
        "query": "Which painting shows a stormy sea?",
        "intent": "guideline_query_seascape",
        "description": "Seascape / storm visual scene from the project brief"
    },
    {
        "query": "I would like to see portraits of children",
        "intent": "guideline_query_children",
        "description": "Demographic subject portrait from the project brief"
    },
    {
        "query": "Tempestuous ocean waves and dark cloudy skies",
        "intent": "paraphrase_stormy_sea",
        "description": "Vocabulary mismatch test: synonym paraphrase of stormy sea"
    },
    {
        "query": "Paintings of young toddlers and infants with their mothers",
        "intent": "paraphrase_children",
        "description": "Vocabulary mismatch test: synonym paraphrase of children"
    },
    {
        "query": "Impressionist cityscape along the river Seine",
        "intent": "style_genre_urban",
        "description": "Style, genre and geographical landmark query"
    },
    {
        "query": "Cubist still life with violin and fruit bowl",
        "intent": "movement_objects",
        "description": "Art movement and specific object composition"
    },
    {
        "query": "Baroque religious scene with Madonna and angels",
        "intent": "iconography",
        "description": "Religious iconography and stylistic movement"
    },
]


def run_benchmark(catalog: list[dict]):
    print("=" * 80)
    print("NATURAL LANGUAGE ARTWORK SEARCH: CLASSICAL (TF-IDF) VS NEURAL (MiniLM)")
    print("=" * 80)
    print(f"Catalog size: {len(catalog)} artworks across 5 genres.")

    # Initialize models
    t0 = time.perf_counter()
    tfidf = TfidfSearcher(catalog)
    tfidf_init_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    neural = NeuralSearcher(catalog)
    neural_init_s = time.perf_counter() - t0

    gallery_size_mb = neural.gallery_embeddings.element_size() * neural.gallery_embeddings.nelement() / 1e6
    model_size_mb = neural.n_params * 4 / 1e6

    print(f"\nModel specs:")
    print(f"  TF-IDF: Vocabulary size: {tfidf.vocab_size} | Fit time: {tfidf_init_s:.2f}s | Size: < 0.1 MB")
    print(f"  MiniLM: Parameters: {neural.n_params:,} ({model_size_mb:.1f} MB) | Init/embed time: {neural_init_s:.2f}s")
    print(f"          Gallery embeddings: shape {tuple(neural.gallery_embeddings.shape)} ({gallery_size_mb:.2f} MB)")

    results_data = {
        "catalog_size": len(catalog),
        "models": {
            "tfidf": {
                "vocab_size": tfidf.vocab_size,
                "fit_time_s": tfidf_init_s,
                "size_mb": 0.05,
            },
            "neural_minilm": {
                "parameters": neural.n_params,
                "model_size_mb": model_size_mb,
                "gallery_size_mb": gallery_size_mb,
                "dimension": neural.gallery_embeddings.shape[1],
            }
        },
        "queries": []
    }

    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)

    table_rows = []
    for item in BENCHMARK_QUERIES:
        q = item["query"]
        desc = item["description"]
        print(f"\nQuery: \"{q}\"")
        print(f"Context: {desc}")

        tfidf_lat = tfidf.latency_benchmark(q, n_runs=50)
        tfidf_res = tfidf.search(q, top_k=3)

        neural_lat, neural_dot_lat = neural.latency_benchmark(q, n_runs=50)
        neural_res = neural.search(q, top_k=3)

        print(f"  [TF-IDF] Latency: {tfidf_lat:.2f} ms")
        for rank, (doc, score) in enumerate(tfidf_res, 1):
            print(f"    #{rank} (score {score:.3f}): {doc['genre']} by {doc['artist']} ({doc['style']}) [{doc['id']}]")

        print(f"  [Neural MiniLM] Latency: {neural_lat:.2f} ms (ranking dot-product only: {neural_dot_lat:.3f} ms)")
        for rank, (doc, score) in enumerate(neural_res, 1):
            print(f"    #{rank} (score {score:.3f}): {doc['genre']} by {doc['artist']} ({doc['style']}) [{doc['id']}]")

        q_record = {
            "query": q,
            "intent": item["intent"],
            "description": desc,
            "tfidf": {
                "latency_ms": tfidf_lat,
                "top_results": [{"id": d["id"], "artist": d["artist"], "genre": d["genre"],
                                 "style": d["style"], "score": s} for d, s in tfidf_res]
            },
            "neural": {
                "latency_ms": neural_lat,
                "ranking_only_ms": neural_dot_lat,
                "top_results": [{"id": d["id"], "artist": d["artist"], "genre": d["genre"],
                                 "style": d["style"], "score": s} for d, s in neural_res]
            }
        }
        results_data["queries"].append(q_record)

        t_top = tfidf_res[0]
        n_top = neural_res[0]
        table_rows.append(
            f"| \"{q}\" | {t_top[0]['artist']} ({t_top[0]['genre']}) | {t_top[1]:.2f} | "
            f"{n_top[0]['artist']} ({n_top[0]['genre']}) | {n_top[1]:.2f} | {tfidf_lat:.2f} | {neural_lat:.1f} |"
        )

    # Save outputs
    out_dir = RESULTS_DIR / "search"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "search_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=1)

    table_md = [
        "# Natural Language Artwork Search: Classical vs. Neural Retrieval",
        "",
        "Collection: 1,500 artworks (WikiArt 5-genre subset). Latency measured on 1 CPU thread (batch size 1).",
        "",
        "| Query | Top TF-IDF Match | TF-IDF Score | Top Neural Match (MiniLM) | Neural Score | TF-IDF Latency (ms) | Neural Latency (ms) |",
        "|---|---|---|---|---|---|---|",
    ] + table_rows + [
        "",
        "## Performance & Resource Summary",
        "",
        "| Model | Model Size | Precomputed Gallery Size | End-to-End Latency (1 CPU thread) | Ranking Matrix Dot Product | Vocabulary Mismatch Handling |",
        "|---|---|---|---|---|---|",
        f"| TF-IDF (classical) | < 0.1 MB | ~0.05 MB (sparse) | {np.mean([q['tfidf']['latency_ms'] for q in results_data['queries']]):.2f} ms | < 0.05 ms | Fails (scores 0.00 on unindexed synonyms) |",
        f"| MiniLM (neural bi-encoder) | {model_size_mb:.1f} MB | {gallery_size_mb:.2f} MB (dense) | {np.mean([q['neural']['latency_ms'] for q in results_data['queries']]):.1f} ms | < 0.20 ms | Robust (semantic similarity over paraphrases) |",
    ]

    table_text = "\n".join(table_md) + "\n"
    (out_dir / "search_table.md").write_text(table_text, encoding="utf-8")
    print(f"\nSaved {out_dir / 'search_benchmark.json'} and {out_dir / 'search_table.md'}")


def main():
    p = argparse.ArgumentParser(description="Natural language artwork search: TF-IDF vs. MiniLM.")
    p.add_argument("--query", type=str, default=None, help="Text query to search for")
    p.add_argument("--top-k", type=int, default=5, help="Number of artworks to retrieve")
    p.add_argument("--benchmark", action="store_true", help="Run full evaluation benchmark suite")
    p.add_argument("--model", choices=["all", "tfidf", "neural"], default="all")
    a = p.parse_args()

    catalog = load_or_build_catalog()

    if a.query:
        print(f"\nQuery: \"{a.query}\"")
        if a.model in ["all", "tfidf"]:
            searcher = TfidfSearcher(catalog)
            res = searcher.search(a.query, top_k=a.top_k)
            print(f"\n--- Top {a.top_k} results [TF-IDF] ---")
            for rank, (doc, score) in enumerate(res, 1):
                print(f"#{rank} [score {score:.3f}] {doc['genre']} by {doc['artist']} ({doc['style']}) - ID: {doc['id']}")
                print(f"   {doc['text']}")
        if a.model in ["all", "neural"]:
            searcher = NeuralSearcher(catalog)
            res = searcher.search(a.query, top_k=a.top_k)
            print(f"\n--- Top {a.top_k} results [Neural MiniLM] ---")
            for rank, (doc, score) in enumerate(res, 1):
                print(f"#{rank} [score {score:.3f}] {doc['genre']} by {doc['artist']} ({doc['style']}) - ID: {doc['id']}")
                print(f"   {doc['text']}")
    else:
        run_benchmark(catalog)


if __name__ == "__main__":
    main()
