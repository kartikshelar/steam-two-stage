# Two-stage Steam recommender

Two-stage game recommender on Steam reviews: retrieval, then ranking, with point-in-time features and offline experiment estimators. Offline machinery only — no production traffic, no real users, no live A/B test.

**Phase: 6 done. Azure port applied, loadgen logged, stack destroyed. Retrieval architecture is frozen. No online experiment was run. No production traffic was served.**

---

## The label experiment (the hook)

Play time is an awkward implicit signal. A 0.3-hour purchase, a 2,000-hour addiction, and a 40-hour negative review are not the same preference. Before any architecture work, this repo treats that as the modeling decision.

Three label formulations, same two-tower retrieval model:

1. **Binary purchase** — ignore hours
2. **Thresholded play time** — positive if `hours > 2.0`
3. **Within-game percentile** — positive if hours ≥ the item's train-only median hours

### Pre-registered prediction (written 2026-09-13, before any training run)

Naive thresholding (`hours > 2.0`) will over-recommend long games relative to binary purchase and relative to within-game percentile labels.

Operationalization: for each model, take the top-10 retrieved items per test user on the **temporal** split, evaluating against the **full catalog** (no sampled negatives), masking train items. Map each item to its median train-set hours (a proxy for typical game length). Average those per-user medians. Predicted order of this statistic:

**thresholded > purchase > percentile**

**The prediction held** on run `20260914T002244Z` (`results/label_experiment_temporal_20260914T002244Z.csv`). Two-tower recommended-game-length (mean of per-user median train hours of top-10 recs, 20,000 temporal test users, full catalog):

| label | two-tower | most-popular | Recall@10 two-tower | Recall@10 most-popular | NDCG@10 two-tower | NDCG@10 most-popular |
|---|---|---|---|---|---|---|
| threshold (`hours > 2`) | **27.42** | 60.99 | 0.0596 | **0.0663** | 0.0513 | **0.0678** |
| purchase | 23.47 | 46.94 | 0.0506 | **0.0570** | 0.0473 | **0.0649** |
| percentile (train-only median) | 20.89 | 47.06 | 0.0479 | **0.0519** | 0.0340 | **0.0432** |

Most-popular is stronger at K=10. That is the headline retrieval number, not a footnote.

---

## Two-stage architecture

```
        user request
             │
             ▼
  ┌─────────────────────┐
  │  STAGE 1: RETRIEVAL │   content two-tower + FAISS HNSW
  │  32,132 games → 500 │   exact inner product is the metric
  └─────────────────────┘
             │
             ▼
  ┌─────────────────────┐
  │  STAGE 2: RANKING   │   PIT GBDT on retrieved 500
  │  500 → ordered list │   exact IP retrieval is frozen
  └─────────────────────┘
             │
             ▼
      POST /recommend (retrieve 500, then PIT GBDT)
```

Frozen at end of Phase 1 (`results/phase1_architecture.json`): user ID tower; item tower is ID (unk for never-trained items) + mean genre + mean tag + price/year, in-batch negatives, logQ correction, ID dropout 0.2. Training label for the frozen model is **purchase**. Changing this architecture invalidates later comparisons.

---

## Results

**Temporal split is the headline. Random split is a labeled inflation comparison only. The most-popular baseline is reported alongside every retrieval number.**

### Phase 1 retrieval (frozen architecture, purchase label)

Run `20260914T072255Z`. Exact inner product over the **32,132-game metadata catalog**. 20,000 eval users per slice. Source: `results/phase1_retrieval_20260914T072255Z.csv`.

| split | model | slice | R@10 | R@50 | R@100 | NDCG@10 |
|---|---|---|---|---|---|---|
| **temporal (headline)** | most-popular | overall | **0.0570** | **0.1521** | 0.2223 | **0.0649** |
| **temporal (headline)** | two-tower + content | overall | 0.0517 | 0.1514 | **0.2347** | 0.0516 |
| temporal | two-tower + content | train catalog only | 0.0601 | 0.1739 | 0.2682 | 0.0623 |
| temporal | most-popular | cold item | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| temporal | two-tower + content | cold item | **0.0048** | 0.0249 | 0.0405 | 0.0027 |
| temporal | most-popular | cold user | **0.0774** | 0.2076 | 0.2869 | **0.0466** |
| temporal | two-tower + content | cold user | 0.0681 | 0.1822 | 0.2694 | 0.0429 |
| random (inflated) | most-popular | overall | 0.0407 | 0.1624 | 0.2590 | 0.0445 |
| random (inflated) | two-tower + content | overall | **0.0880** | 0.2400 | 0.3433 | 0.0770 |

On the temporal full catalog, most-popular still wins Recall@10. Restricting the content model to the train catalog (the Phase 0 setting) it does beat most-popular at Recall@10 (0.0601 vs 0.0570) and beats the Phase 0 ID two-tower (0.0506). Random split inflates the content model's Recall@10 from 0.0517 to 0.0880 — 1.70× — and would reverse the K=10 conclusion. That is why temporal is the headline.

**Popularity-decile Recall@10 (temporal, purchase, full catalog).** Most-popular is 0.0 on deciles 1–9 and 0.0837 on decile 10. Content two-tower is 0 / 0 / 0.0007 / 0.0009 / 0.0020 / 0.0031 / 0.0027 / 0.0067 / 0.0097 / 0.0724. Still almost entirely the head.

ANN (FAISS HNSW, 32,132 × 64, `results/ann_temporal_20260914T072255Z.json`): overlap with exact top-500 is 0.684; single-query p50 0.28 ms. Headline metrics above are exact search, not ANN.

### Phase 3 two-stage ranking (frozen retrieval + PIT GBDT)

Run `20260914T084052Z`. Temporal split, purchase label, **full catalog**. Retrieve 500 with the frozen content two-tower (exact inner product), rerank those 500 with a LightGBM trained on PIT features plus `retrieval_score`. Same 20,000 eval users as the retrieval numbers above. Source: `results/phase3_ranking_20260914T084052Z.csv`.

This is not sampled-negative ranking. The ranker cannot recover a test item the retriever never returned. Retrieval Recall@500 (the ceiling) is 0.5267 overall.

| slice | model | R@10 | R@50 | R@100 | NDCG@10 |
|---|---|---|---|---|---|
| **overall (headline)** | most-popular | 0.0570 | 0.1521 | 0.2223 | **0.0649** |
| **overall (headline)** | two-tower + content | 0.0517 | 0.1514 | 0.2347 | 0.0516 |
| **overall (headline)** | two-tower then GBDT | **0.0764** | **0.2291** | **0.3253** | 0.0644 |
| cold user | most-popular | 0.0774 | 0.2076 | 0.2869 | 0.0466 |
| cold user | two-tower + content | 0.0681 | 0.1822 | 0.2694 | 0.0429 |
| cold user | two-tower then GBDT | **0.0977** | **0.2497** | **0.3417** | **0.0619** |
| cold item | most-popular | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| cold item | two-tower + content | 0.0048 | 0.0249 | **0.0405** | 0.0027 |
| cold item | two-tower then GBDT | **0.0110** | **0.0278** | 0.0376 | **0.0070** |

Two-stage is the first model in this repo to beat most-popular on temporal Recall@10. NDCG@10 is a tie with most-popular (0.0644 vs 0.0649). Top features: `item_n`, then `retrieval_score`, then `item_n_7d` (`results/gbdt_ranker_importance.csv`). Weakest: `item_is_free`, `ui_n`, `item_log_price`. No feature had zero gain.

**Popularity-decile Recall@10 (two-stage).** 0.0016 / 0.0022 / 0.0082 / 0.0040 / 0.0039 / 0.0122 / 0.0144 / 0.0081 / 0.0164 / 0.1022. Non-zero on the tail; still mostly the head.

### Phase 0 label experiment (ID two-tower, three labels)

Logged run: `20260914T002244Z`. Train catalog. Source: `results/label_experiment_temporal_20260914T002244Z.csv`.

| label | model | R@10 | R@50 | R@100 | NDCG@10 | NDCG@50 | NDCG@100 |
|---|---|---|---|---|---|---|---|
| purchase | most-popular | **0.0570** | 0.1521 | 0.2223 | **0.0649** | 0.0856 | 0.1036 |
| purchase | two-tower | 0.0506 | **0.1588** | **0.2445** | 0.0473 | 0.0779 | 0.0994 |
| threshold | most-popular | **0.0663** | 0.1743 | 0.2583 | **0.0678** | 0.0938 | 0.1144 |
| threshold | two-tower | 0.0596 | **0.1812** | **0.2748** | 0.0513 | 0.0859 | 0.1085 |
| percentile | most-popular | **0.0519** | 0.1364 | 0.2010 | **0.0432** | 0.0648 | 0.0792 |
| percentile | two-tower | 0.0479 | **0.1437** | **0.2185** | 0.0340 | 0.0602 | 0.0765 |

---

## Point-in-time leak gap

Run `20260914T082225Z`. Same `(user, item, t, label)` rows for both models. LightGBM binary, purchase label, temporal split. **This is a leak diagnostic on sampled candidates (1 positive + 10 random catalog negatives per test event). It is not a retrieval number.** Retrieval still evaluates against the full 32,132-game catalog in Phase 1.

Source: `results/leak_gap_20260914T082225Z.csv`. 250,000 train positives / 4 negatives; 40,000 test positives / 10 negatives; 27,210 test users. `features/test_no_leakage.py` passes (24 tests). 200 real-data PIT test positives matched a strict `ts < t` count.

| model | AUC | AP | NDCG@10 |
|---|---|---|---|
| PIT (correct) | 0.9722 | 0.7927 | 0.9215 |
| leaked (wrong on purpose) | **0.9999** | **0.9993** | **0.9997** |
| gap (leaked − PIT) | 0.0277 | **0.2066** | 0.0782 |

The AUC gap is small because ranking one purchase against 10 random catalog games is already easy with recent item popularity (`item_n_30d` is the PIT top feature). Average precision is the honest gap: leak lifts AP from 0.7927 to 0.9993.

Leaked top feature: `ui_n` (includes the current review and any later ones). PIT `ui_n` gain is **0.0** — last-review-per-pair plus emit-before-observe means the pair has never been seen at prediction time. Importances: `results/gbdt_pit_importance.csv`, `results/gbdt_leaked_importance.csv`.

---

## Serving latency

**Not production traffic.** Timed `POST /recommend` (retrieve 500, PIT GBDT rerank, return k=20) against a short-lived FastAPI endpoint. Headline hardware is AWS. Source: `results/serving_latency.csv`.

| hardware | concurrency | requests | errors | p50 ms | p95 ms | p99 ms | run |
|---|---|---|---|---|---|---|---|
| **aws_t3.medium (headline)** | 8 | 200 | 0 | **194.7** | **275.3** | **302.7** | `20260915T004053Z` |
| azure_Standard_B2s_v2 (port) | 8 | 200 | 0 | 129.8 | 174.5 | 228.2 | `20260916T085546Z` |
| local_cpu (comparison) | 8 | 200 | 0 | 159.7 | 218.9 | 290.1 | `20260915T002828Z` |

Training-serving skew: 200 sampled pairs, 0 mismatches (`results/skew_check.json`, `20260915T002631Z`). Serving calls the same `PITState.features` function the ranker was trained on. AWS stack was destroyed after that run (`terraform destroy` in `terraform/aws`). Azure is a port of the same flags (budget first, public subnet, no NAT, one CPU VM). `Standard_B2s` had no capacity in `eastus` or `westus2`; the logged Azure row is `Standard_B2s_v2` in `westus2` through a Standard Load Balancer (`20260916T085546Z`). Azure stack is destroyed after that run (`terraform destroy` in `terraform/azure`). Neither session has a Cost Explorer / Cost Management extract; both were minutes of one CPU VM plus a load balancer, under the $10 ceiling. Not production traffic.

---

## Experimentation

**No online experiment was run. There are no real users and no live feedback loop.** Run `20260914T212349Z` builds the offline machinery on 10,000 temporal test users: simulated two-tower rank-softmax logs, IPS/SNIPS for the GBDT policy, team-draft interleaving with held-out purchases as clicks, and an MDE calculator. Source: `results/phase4_ope_20260914T212349Z.json`.

Logging policy = rank-softmax of the frozen two-tower over its retrieved 500. Target = rank-softmax of the PIT GBDT over the same 500. Reward = 1 if the sampled item is a held-out purchase. Clip = 20. Temperature = 1.0.

| estimator | logging policy (two-tower) | GBDT target |
|---|---|---|
| on-policy MC (simulated) | 0.0587 | 0.0619 |
| IPS (from two-tower logs) | 0.0587 (self-check) | **0.0083** |
| SNIPS (from two-tower logs) | 0.0587 (self-check) | 0.0799 |
| IPS ESS | 10,000 | **191** |

IPS of the logging policy on its own logs recovers MC (0.0587). IPS of the GBDT policy does not (0.0083 vs MC 0.0619). Unclipped weights hit 1,097. Bootstrap 95% CI for IPS is [0.0048, 0.0135]; for SNIPS [0.0482, 0.1289]. SNIPS is less biased and too noisy to call a winner.

Team-draft interleave @10 (A vs B, GT items as clicks, 10,000 users):

| A vs B | A wins | B wins | ties | P(A wins \| decisive) |
|---|---|---|---|---|
| most-popular vs GBDT | **1,669** | 1,118 | 7,213 | 0.599 |
| two-tower vs GBDT | 1,210 | 1,263 | 7,527 | 0.489 (p=0.29) |

Most-popular takes more interleaved credit than GBDT even though GBDT wins Recall@10 (0.0786 vs 0.0603) and hit@10 (0.2923 vs 0.2630) on these same users. Most-popular P@1 is 0.1041 vs GBDT 0.0719 — the head item is where team-draft pays.

**MDE** (two-sided two-proportion z-test, α=0.05, power=0.8, 50/50, one session per user per day). Baseline = most-popular hit@10 = 0.263 on this sample. **DAU figures are hypothetical storefront assumptions. They are not Steam traffic and not this dataset.**

| relative lift | abs. lift | n per arm | days @ 10k DAU | days @ 100k DAU | days @ 1M DAU |
|---|---|---|---|---|---|
| 1% | 0.00263 | 441,303 | **89** | 9 | 1 |
| 2% | 0.00526 | 110,674 | 23 | 3 | 1 |
| 5% | 0.0132 | 17,871 | 4 | 1 | 1 |
| 10% | 0.0263 | 4,533 | 1 | 1 | 1 |
| 11.1% (observed hit@10 GBDT vs pop) | 0.0293 | 3,664 | 1 | 1 | 1 |

A 1% relative move on this metric is not cheap at 10k DAU. The 11% offline hit@10 lift would show in a day *if* it transferred online. IPS did not recover that lift from two-tower logs, so this calculator is not a reason to skip an experiment. It is a reason to size one.

---

## What didn't work

1. **Retrieval still loses temporal Recall@10 to most-popular.** Content two-tower 0.0517 vs 0.0570. Ranking is what flips K=10: two-stage 0.0764 (`20260914T084052Z`).
2. **NDCG@10 does not beat most-popular.** Two-stage 0.0644 vs most-popular 0.0649. More hits, not better position of the first hit versus a popularity list.
3. **Cold-user retrieval loses to most-popular.** Mean user embedding, Recall@10 0.0681 vs 0.0774. The ranker then wins that slice (0.0977) using item popularity and the retrieval score, not a user content tower.
4. **Cold-item recall is still tiny.** Two-stage 0.0110 at K=10 vs retrieval 0.0048 vs most-popular 0. At K=100 the ranker is slightly worse than retrieval (0.0376 vs 0.0405) — it can bury cold games that made the 500.
5. **The retriever is the ceiling.** Only 52.7% of overall test positives are in the top 500. 16.3% of train positives were missing from the 500 and had to be injected for training.
6. **HNSW overlap with exact top-500 is 0.684.** Usable for a latency check (p50 0.28 ms). Not used for reported Recall@K — those are exact. At 32k items, exact search is already cheap.
7. **The ID two-tower (Phase 0) does not beat most-popular at K=10** on the train catalog either (0.0506 vs 0.0570).
8. **Almost all overall recall is still the head.** Two-stage Recall@10 on decile 10 is 0.1022; deciles 1–9 stay at or below 0.0164. Ranking puts a few tail items in the top 10 (deciles 1–2 are no longer exactly 0) and does not fix the head bias.
9. **Most-popular over-recommends long games even more than thresholded two-tower.** Thresholded most-popular game length is 60.99 hours vs 27.42 for the thresholded two-tower.
10. **A leaked ranker looks solved.** AUC 0.9999 / AP 0.9993 on the same sampled candidates where PIT is 0.9722 / 0.7927 (`20260914T082225Z`). The leaked model’s top feature is `ui_n`.
11. **PIT `ui_n` barely helps ranking.** Gain is the second-lowest on the Phase 3 model. Last-review-per-pair means the pair is unseen at `t`.
12. **`item_is_free` and `item_log_price` add almost nothing** once `item_n` and `item_price` are in the tree.
13. **PIT AUC 0.9722 is not a retrieval win.** It is popularity vs 10 random catalog items. Do not compare it to Recall@10.
14. **IPS does not recover the GBDT policy from two-tower logs.** Estimate 0.0083 vs on-policy MC 0.0619. ESS 191 / 10,000. Unclipped weights to 1,097 (`20260914T212349Z`).
15. **Team-draft disagrees with Recall@10.** Most-popular beats GBDT on decisive interleaved slates (1,669 vs 1,118) while losing Recall@10 and hit@10 on the same 10,000 users. P@1 is the difference (0.1041 vs 0.0719).

The Phase 0 pre-registration was not falsified: thresholded two-tower recs are longer than purchase recs, which are longer than percentile recs (27.42 > 23.47 > 20.89).

---

## What this still cannot claim

Be honest in interviews. Overclaiming here is worse than the gap.

- No production recsys experience
- No live feedback loop, so no experience with degenerate loops or position bias in a deployed system
- No real A/B test
- No real-time feature serving
- No exposure to the organizational side (metric selection, guardrail metrics, launch decisions)
- No Azure production serving — one timed loadgen on a short-lived `Standard_B2s_v2`, then destroy

The honest framing: built the offline machinery and can reason about the problems. Has not operated a recommender in production.

---

## What was not built (and will not be)

- No online experiment, no real users, no live feedback loop, no production recsys experience
- No sequence models (SASRec, BERT4Rec)
- No LLM / review-text embeddings
- No UI
- No real-time streaming features
- No general-purpose feature store
- No second recsys stack on Azure. The Azure session reused the same serving image and PIT bundle as AWS.

---

## Dataset

McAuley lab Steam reviews (UCSD), Version 2.

- Source page: https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data
- Files: `steam_reviews.json.gz`, `steam_games.json.gz`, `bundle_data.json.gz`
- Published counts on that page: 7,793,069 reviews, 2,567,538 users, 15,474 items, 615 bundles
- Measured from this dump (`results/dataset_profile.json`): 7,793,069 gzip records; **6,889,728** unique `(user, item)` pairs; **2,567,538** users; **15,474** items in reviews; **32,132** rows in game metadata (SASRec reported 32,135). 16,658 metadata-only games have no review
- License: no OSI license is published. The lab asks that the data be used for research and cited. The dump is **not** redistributed in this repository
- Schema (reviews): `username`, `hours`, `products`, `product_id`, `date`, `text`, `early_access`, …
- Schema (games): `id`, `title` / `app_name`, `genres`, `tags`, `price`, `release_date`, `publisher`, `developer`, …

Cite:

- Kang & McAuley, *Self-attentive sequential recommendation*, ICDM 2018
- Wan & McAuley, *Item recommendation on monotonic behavior chains*, RecSys 2018
- Pathak, Gupta, McAuley, *Generating and personalizing bundle recommendations on Steam*, SIGIR 2017

---

## Reproduction

Python 3.9+. NVIDIA GPU optional (a 4 GB GTX 1650 is enough for the ID two-tower after the train-only 5-core).

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# GPU (CUDA 11.8 wheels; skip for CPU):
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118

python -m pytest
python data/prepare.py
python data/splits.py
python data/labels.py
python experiments/label_experiment.py --split temporal
python experiments/phase1_retrieval.py --split both
python experiments/leak_gap.py
python experiments/phase3_ranking.py
python experiments/phase4_ope.py
python serving/export_bundle.py
python serving/skew_check.py
python -m uvicorn serving.app:app --host 127.0.0.1 --port 8000
# other terminal:
python serving/loadgen.py --base-url http://127.0.0.1:8000 --hardware local_cpu
```

AWS (budget first; see `docs/aws-budget-guardrails.md`):

```powershell
cd terraform\aws
copy terraform.tfvars.example terraform.tfvars
# set alert_email; leave enable_network/enable_compute false
terraform init
terraform apply
# confirm SNS email, then enable_network, apply, publish image, enable_compute, loadgen, terraform destroy
```

Azure (port of the AWS stack; budget first; RSA SSH key; see `docs/azure-port.md`):

```powershell
az login
az account show   # must succeed
cd terraform\azure
copy terraform.tfvars.example terraform.tfvars
# set alert_email; leave enable_network/enable_compute false, instance_count 0
terraform init
terraform apply
# confirm budget email, then enable_network, publish to ACR, enable_compute, instance_count=1, loadgen, terraform destroy
```

Or: `python run_phase0.py` then `python run_phase1.py` then `python run_phase2.py` then `python run_phase3.py` then `python run_phase4.py`.

Smoke pipeline (not a headline run; label any numbers it produces as smoke):

```powershell
python run_phase0.py --config configs/phase0_smoke.yaml --force-splits
```

Splits are frozen in `data/splits/manifest.json`. Rebuilding requires `--force`.

---

## Phase status

| Phase | Status |
|---|---|
| 0 Data and labels | **done** (run `20260914T002244Z`) |
| 1 Retrieval | **done** (run `20260914T072255Z`; architecture frozen) |
| 2 Point-in-time features | **done** (run `20260914T082225Z`) |
| 3 Ranking | **done** (run `20260914T084052Z`) |
| 4 Experimentation layer | **done** (run `20260914T212349Z`) |
| 5 Serving | **done** (AWS run `20260915T004053Z`; stack destroyed) |
| 6 Write-up and Azure port | **done** (Azure run `20260916T085546Z`; stack destroyed) |
