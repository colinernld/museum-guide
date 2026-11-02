"""Extra checks behind the search slide of the defence, on top of the benchmark of src/search.py.

Reuses the searchers of src/search.py without changing them or their results:
  - a query whose word is in no description ("a shipwreck");
  - the measured size of the TF-IDF index;
  - the latency of the 7 benchmark queries on this machine (1 thread, batch 1).
"""
import json
import pickle
import platform
import subprocess

import numpy as np

from data import out_path
from search import BENCHMARK_QUERIES, NeuralSearcher, TfidfSearcher, load_or_build_catalog

ABSENT_WORD_QUERY = "a shipwreck"


def cpu_name():
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or platform.machine()


def main():
    catalog = load_or_build_catalog()
    tfidf = TfidfSearcher(catalog)
    neural = NeuralSearcher(catalog)

    # Word in no description: TF-IDF only returns artworks with a score of 0, so it has no match
    in_text = sum(ABSENT_WORD_QUERY.split()[-1] in d["text"].lower() for d in catalog)
    tfidf_best = max(score for _, score in tfidf.search(ABSENT_WORD_QUERY, top_k=10))
    neural_top = neural.search(ABSENT_WORD_QUERY, top_k=10)
    neural_artists = sorted({d["artist"] for d, _ in neural_top})
    print(f'"{ABSENT_WORD_QUERY}": in {in_text} descriptions')
    print(f"  TF-IDF best score: {tfidf_best:.3f}")
    print(f"  MiniLM top 10 artists: {', '.join(neural_artists)} (best score {neural_top[0][1]:.3f})")

    # Size of the TF-IDF index, measured
    m = tfidf.doc_matrix
    index_mb = (m.data.nbytes + m.indices.nbytes + m.indptr.nbytes) / 1e6
    vectorizer_mb = len(pickle.dumps(tfidf.vectorizer)) / 1e6
    print(f"TF-IDF size: vectorizer {vectorizer_mb:.3f} MB + index {index_mb:.3f} MB")

    # Latency on this machine
    cpu = cpu_name()
    queries = [item["query"] for item in BENCHMARK_QUERIES]
    tfidf_ms = [tfidf.latency_benchmark(q, n_runs=50) for q in queries]
    neural_ms = [neural.latency_benchmark(q, n_runs=50)[0] for q in queries]
    print(f"Latency on {cpu}, 1 thread, {len(queries)} queries:")
    print(f"  TF-IDF mean {np.mean(tfidf_ms):.2f} ms (from {min(tfidf_ms):.2f} to {max(tfidf_ms):.2f})")
    print(f"  MiniLM mean {np.mean(neural_ms):.2f} ms (from {min(neural_ms):.2f} to {max(neural_ms):.2f})")

    out = {
        "absent_word_query": {
            "query": ABSENT_WORD_QUERY,
            "descriptions_with_word": in_text,
            "tfidf_best_score": tfidf_best,
            "neural_top10_artists": neural_artists,
            "neural_best_score": neural_top[0][1],
        },
        "tfidf_size_mb": {"vectorizer": vectorizer_mb, "index": index_mb},
        "latency": {"cpu": cpu, "tfidf_ms": tfidf_ms, "neural_ms": neural_ms},
    }
    path = out_path("search", "search_checks.json")
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Saved {path.name}")


if __name__ == "__main__":
    main()
