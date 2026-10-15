# Museum guide: artwork style classification

Neural-network components of an intelligent museum guide (WikiArt subset).

## Reproduce the data
    pip install -r requirements.txt
    python make_wikiart_subset.py --label style \
        --classes Impressionism Baroque Cubism Ukiyo_e Pop_Art \
        --per-class 300 --size 128 --out data/wikiart_style5

Images are not committed (licence unknown). See subset_info.json for the exact command.
