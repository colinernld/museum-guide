# Natural Language Artwork Search: Classical vs. Neural Retrieval

Collection: 1,500 artworks (WikiArt 5-genre subset). Latency measured on 1 CPU thread (batch size 1).

| Query | Top TF-IDF Match | TF-IDF Score | Top Neural Match (MiniLM) | Neural Score | TF-IDF Latency (ms) | Neural Latency (ms) |
|---|---|---|---|---|---|---|
| "Which painting shows a stormy sea?" | William Turner (cityscape) | 0.29 | William Turner (cityscape) | 0.48 | 0.18 | 1.8 |
| "I would like to see portraits of children" | Zinaida Serebriakova (still_life) | 0.32 | Zinaida Serebriakova (portrait) | 0.50 | 0.18 | 1.8 |
| "Tempestuous ocean waves and dark cloudy skies" | William Turner (cityscape) | 0.26 | William Turner (cityscape) | 0.46 | 0.18 | 1.8 |
| "Paintings of young toddlers and infants with their mothers" | Mary Cassatt (portrait) | 0.32 | Unknown Artist (religious_painting) | 0.52 | 0.18 | 1.9 |
| "Impressionist cityscape along the river Seine" | Claude Monet (cityscape) | 0.40 | Claude Monet (cityscape) | 0.76 | 0.18 | 1.9 |
| "Cubist still life with violin and fruit bowl" | Juan Gris (still_life) | 0.29 | Georges Braque (still_life) | 0.69 | 0.18 | 1.9 |
| "Baroque religious scene with Madonna and angels" | Bartolome Esteban Murillo (religious_painting) | 0.38 | Bartolome Esteban Murillo (religious_painting) | 0.70 | 0.18 | 1.8 |

## Performance & Resource Summary

| Model | Model Size | Precomputed Gallery Size | End-to-End Latency (1 CPU thread) | Ranking Matrix Dot Product | Vocabulary Mismatch Handling |
|---|---|---|---|---|---|
| TF-IDF (classical) | < 0.1 MB | ~0.05 MB (sparse) | 0.18 ms | < 0.05 ms | Fails (scores 0.00 on unindexed synonyms) |
| MiniLM (neural bi-encoder) | 90.9 MB | 2.30 MB (dense) | 1.8 ms | < 0.20 ms | Robust (semantic similarity over paraphrases) |
