# Data

This repository uses the Amazon Reviews 2014 5-core datasets:

- Beauty
- Sports and Outdoors
- Toys and Games

The raw files are downloaded from the public SNAP Amazon product graph dataset. This anonymous release does not redistribute raw data, processed data, train/eval splits, generated embeddings, checkpoints, or model outputs.

Example:

```bash
python data/download_amazon_dataset.py --category Beauty --type both
```

The preprocessing pipeline constructs:

- chronological user sequences
- item metadata text
- dense item embeddings
- four-layer Semantic IDs
- SID cognitive-map edges
- difficulty labels
- supervised training and evaluation files

Run the offline preprocessing stages with:

```bash
python run_generate_data.py --category Beauty --phase download
python run_generate_data.py --category Beauty --phase sid
python run_generate_data.py --category Beauty --phase hnsw
```

Use `--category Sports`, `--category Toys`, or `--category all` for other settings.

