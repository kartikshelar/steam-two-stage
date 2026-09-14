# Dataset notes

Figures in the project brief (~7.79M reviews, ~2.57M users, ~32,135 games) are **not** copied into results. Re-measure from the files.

## Source

- Lab page: https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data
- Canonical files: https://mcauleylab.ucsd.edu/public_datasets/data/steam/
  - `steam_reviews.json.gz` (~1.2–1.3 GB) — Version 2 reviews
  - `steam_games.json.gz` (~2.5–2.7 MB) — item metadata
  - `bundle_data.json.gz` (~90 KB)
- HuggingFace mirror (fallback): https://huggingface.co/datasets/recommender-system/steam-review-and-bundle-dataset

The lab documents the parser as `eval(l)` over gzip lines ("loose json"). `data/prepare.py` tries `json.loads`, then `ast.literal_eval`, then a builtins-empty `eval`.

## Published counts (for comparison, not for reporting)

From the McAuley page (retrieved 2026-09-13):

| quantity | published |
|---|---|
| Reviews | 7,793,069 |
| Users | 2,567,538 |
| Items | 15,474 |
| Bundles | 615 |

Kang's SASRec README reports 32,135 games for the metadata file. HuggingFace's copy of the same dump also says 32,135 Steam games. Those are metadata rows, not necessarily items that appear in reviews. `results/dataset_profile.json` records both after parse.

## License

No OSI license is posted on the lab page. Use is research-oriented and requires citation. Do not vendor the dump in git.

## Schema (Version 2 review example)

```
username, hours, products, product_id, page_order, date, text, early_access, page
```

User identity is a **display name**, not a Steam64 ID. Collisions are possible. That limitation stays in the write-up.

Hours are play hours at review time (float). Dates are calendar strings (`YYYY-MM-DD`).

## Why this dump and not another

Play time is a non-binary, non-linear implicit label. That awkwardness is the point of Phase 0. MovieLens, Amazon stars, and playlist dumps do not pose the same label problem.
