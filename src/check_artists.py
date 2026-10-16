"""Artist leakage check: how many validation/test paintings are by a painter also seen in training?

A random split can put works of the same painter in train and test; the model may then
recognise the painter's hand rather than the genre, and test accuracy is optimistic.
"Unknown Artist" is a catch-all label, not one painter, so it is reported separately.
Compares the original random split with the artist-disjoint split now used everywhere.
"""
import json

import numpy as np
import pandas as pd

from data import UNKNOWN_ARTIST, artists, get_splits, out_path

report = {}
for by_artist in [False, True]:
    name = "artist_split" if by_artist else "random_split"
    ds, tr, va, te = get_splits(by_artist=by_artist)
    art = artists(ds)
    named = art != UNKNOWN_ARTIST
    train_artists = set(art[tr][named[tr]])
    report[name] = {}
    print(f"== {name} ==")
    for split, idx in [("val", va), ("test", te)]:
        a_named = art[idx][named[idx]]
        seen = int(np.isin(a_named, list(train_artists)).sum())
        report[name][split] = {"images": len(idx), "by_named_painter": len(a_named),
                               "painter_also_in_train": seen,
                               "share_of_split": round(seen / len(idx), 3)}
        print(f"  {split:5s}: {len(idx)} images, {len(a_named)} by a named painter, "
              f"{seen} by a painter also in train ({seen / len(idx):.0%} of the split)")

report["unknown_artist_images"] = int((~named).sum())
report["named_painters"] = len(set(art[named]))
top = pd.Series(art[named]).value_counts().head(10)
report["top_painters"] = {k: int(v) for k, v in top.items()}
print(f"\n'Unknown Artist': {(~named).sum()} images | {report['named_painters']} named painters")
print("Most frequent named painters:\n" + top.to_string())
json.dump(report, open(out_path("data", "artist_leakage.json"), "w"), indent=1)
print("Saved results/data/artist_leakage.json")
