# Museum guide: artwork style classification

Neural-network components of an intelligent museum guide (WikiArt subset).

## Reproduce the data
    pip install -r requirements.txt
    python make_wikiart_subset.py --label genre \
        --classes portrait landscape cityscape still_life religious_painting \
        --per-class 300 --size 128 --out data/wikiart_genre5

Images are not committed (licence unknown). See subset_info.json for the exact command.
