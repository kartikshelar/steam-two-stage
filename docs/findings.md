# Findings

Per-phase analysis. Negative results get equal billing. Numbers that are not in `results/` do not belong here.

---

## Phase 0 — Data and labels

### Pre-registered prediction (2026-09-13, before any two-tower training run)

Naive thresholding of play time (`hours > 2.0`) will over-recommend long games relative to binary purchase and relative to within-game percentile labels.

Operationalization: mean over temporal-split test users of the median train-set hours of their top-10 retrieved items (full catalog, train items masked). Predicted order:

**thresholded > purchase > percentile**

Logged in `results/pre_registration.json` before `experiments/label_experiment.py` started training.

### Dataset profile

Source: `results/dataset_profile.json` (run `2026-09-14T00:09:17Z`).

The gzip contains 7,793,069 records, matching the McAuley page. After keeping the last review per `(user_id, item_id)`, **6,889,728** unique pairs remain.

| quantity | this dump | McAuley page | SASRec README |
|---|---|---|---|
| review records (gzip) | 7,793,069 | 7,793,069 | 7,793,069 |
| unique user–item pairs | 6,889,728 | — | — |
| users | 2,567,538 | 2,567,538 | 2,567,538 |
| items in reviews | 15,474 | 15,474 | — |
| items in `steam_games.json.gz` | 32,132 | — | 32,135 |

The 32k vs 15k gap is real: 16,658 metadata rows have no review in this dump. Every reviewed item has metadata.

Other logged facts:

- Date range: 2010-10-15 to 2018-01-05
- Hours median 13.6, p90 196.7, p99 1,588, max 42,100.7. 17.1% of reviews have hours < 2
- Median reviews per user: 1. **89.0% of users have fewer than 5 reviews.** Cold start is not a toy slice
- Median reviews per item: 32. Max: 121,329 (item `440`). Popularity bias is extreme

### Split

Source: `results/split_counts.csv` and `data/splits/manifest.json`.

Temporal cutoff (80th percentile of timestamps): **2017-06-27**. Train-only 5-core then applied to test.

| split | n_train | n_test (warm) | users train | items train | temporally ordered |
|---|---|---|---|---|---|
| temporal (headline) | 2,686,568 | 323,744 | 221,870 | 10,027 | yes |
| random (inflation comparison) | 2,554,102 | 573,461 | 216,491 | 11,287 | no |

Random train and test both span 2010-10-15 through 2018-01-05. That overlap is the inflation.

### Labels

Source: `results/label_stats.csv`. Threshold `hours > 2.0` and per-game train-only median, as pre-registered.

Temporal train positive rates: purchase 100%, threshold 79.6%, percentile 50.4%.
Train-only per-game hour cutoffs: median 3.5h, p90 15.6h (`results/label_stats.json`). A global 2.0h threshold is below most games' own median, so it still treats a lot of short-for-that-title sessions as positives.

### Label experiment

Run `20260914T002244Z`. Source: `results/label_experiment_temporal_20260914T002244Z.csv`. Temporal split, full catalog, 20,000 eval users. Pre-registration file was on disk before this training run.

**Pre-registration held** for the two-tower recommended-game-length statistic: threshold 27.42 > purchase 23.47 > percentile 20.89.

Most-popular on the same metric: threshold 60.99, purchase 46.94, percentile 47.06. Thresholding still lengthens the popular set; purchase vs percentile barely moves it.

**Two-tower vs most-popular (purchase label, temporal):**

| model | R@10 | R@50 | R@100 | NDCG@10 |
|---|---|---|---|---|
| most-popular | 0.0570 | 0.1521 | 0.2223 | 0.0649 |
| two-tower | 0.0506 | 0.1588 | 0.2445 | 0.0473 |

The two-tower loses at K=10 and wins at K=50/100. Same pattern on the other two labels. Negative result, equal billing.

Most-popular Recall@10 is 0.0 on popularity deciles 1–9 (purchase label) and 0.0837 on decile 10. Two-tower tail recall is non-zero and still tiny (decile 1–2 are 0.0 at K=10).

Random-split *model* numbers were not logged in this run. Split counts for random are in `results/split_counts.csv` and are not temporally ordered.

---

## Phase 1 — Retrieval

Run `20260914T072255Z`. Source: `results/phase1_retrieval_20260914T072255Z.csv`. Architecture frozen in `results/phase1_architecture.json`. Purchase label.

Item tower: ID (unk if the item never appeared in train) + mean genre + mean tag + price/year, ID dropout 0.2. Catalog for serving/eval: 32,132 metadata games. Metrics are exact inner product; FAISS HNSW is a serving index.

**Temporal overall (32k catalog):** content two-tower Recall@10 0.0517 vs most-popular 0.0570. Most-popular still wins at K=10. Content wins at K=100 (0.2347 vs 0.2223).

**Matched train catalog** (the Phase 0 protocol): content Recall@10 0.0601, which beats most-popular 0.0570 and the Phase 0 ID two-tower 0.0506. Adding content helps when the catalog is the same. Expanding the catalog to cold items gives those items a chance and costs overall Recall@10.

**Cold items (<5 train interactions):** most-popular Recall@10 is 0. Content is 0.0048 / 0.0249 / 0.0405 at K=10/50/100. The cold-start path is real and weak.

**Cold users (<5 train interactions):** mean user embedding Recall@10 0.0681 vs most-popular 0.0774. No user-side content. Popularity wins.

**Random split (labeled inflation):** content Recall@10 0.0880 vs temporal 0.0517 (1.70×). On the random split the content model beats most-popular at K=10 (0.0880 vs 0.0407). That reversal is why random is not the headline.

**ANN:** FAISS HNSW, 32,132 vectors, overlap@500 with exact = 0.684, single-query p50 = 0.28 ms (`results/ann_temporal_20260914T072255Z.json`). At this catalog size exact search is already cheap; the index exists because the brief requires ANN, not because 32k needs it.

**Popularity deciles:** still a head model. Content Recall@10 on deciles 1–9 stays below 0.01; decile 10 is 0.0724 vs most-popular 0.0837.

---

## Phase 2 — Point-in-time features

Run `20260914T082225Z`. Source: `results/leak_gap_20260914T082225Z.json`.

**Protocol.** Sweep reviews in timestamp order. Emit PIT features from `PITState` *before* `observe` (strict `ts < t`). Leaked features use full-timeline counts and two-sided windows that include now and the future. Same `(user, item, t, label)` rows for both LightGBM models. Train: 250,000 positives + 4 random catalog negatives. Test: 40,000 positives + 10 random catalog negatives. Temporal split, purchase label.

**These AUC/NDCG numbers are the leak comparison, not retrieval.** Retrieval still uses the full 32,132-game catalog. Ranking one held-out purchase against 10 random catalog games is a different task.

**Leak gates.** `features/test_no_leakage.py` (synthetic) plus 200 real PIT test positives whose `item_n` / `user_n` matched `bisect` counts of timestamps strictly before `t` (`n_leak_checks`: 200).

| model | AUC | AP | NDCG@10 | NDCG@50 |
|---|---|---|---|---|
| PIT | 0.9722 | 0.7927 | 0.9215 | 0.9240 |
| leaked | 0.9999 | 0.9993 | 0.9997 | 0.9998 |
| gap (leaked − PIT) | 0.0277 | 0.2066 | 0.0782 | — |

**What the leak buys.** Average precision moves 0.2066. AUC barely moves because PIT is already 0.9722 — recent item counts separate a real purchase from a random catalog draw. The leaked model is essentially perfect.

**Feature importances** (`results/gbdt_pit_importance.csv`, `results/gbdt_leaked_importance.csv`):

- PIT top feature: `item_n_30d` (then `item_n_90d`, `item_n`). Recent popularity.
- Leaked top feature: `ui_n`. That count includes the current review (and any later ones). Smoking gun.
- PIT `ui_n` gain is 0.0. Last-review-per-pair plus emit-before-observe means the pair is unseen at `t`.
- `user_days_since_last` gain is 0.0 on the leaked model: global last-timestamp is at or after `t`, so recency collapses.

**What this does not say.** It does not say ranking will beat retrieval on the full catalog. It does not say the two-tower is fixed. It says a naive groupby leak produces a leaderboard number you cannot serve, and that the honest PIT model on this candidate mix is mostly a 30-day popularity ranker.

---

## Phase 3 — Ranking

Run `20260914T084052Z`. Source: `results/phase3_ranking_20260914T084052Z.json`. Frozen Phase 1 two-tower. PIT features as of the temporal cutoff (`2017-06-27`). LightGBM trained on 10,000 train queries × (1 positive + 50 negatives sampled from that query’s retrieved 500). Eval reranks all 500. 25 PIT `item_n` checks at cutoff matched strict pre-cutoff counts.

**Headline (temporal, full catalog, 20,000 users, purchase).** Two-stage Recall@10 **0.0764** vs most-popular **0.0570** vs retrieval-only **0.0517**. First model in this repo to beat most-popular at K=10. Recall@50 / @100: 0.2291 / 0.3253 vs retrieval 0.1514 / 0.2347 vs pop 0.1521 / 0.2223.

**NDCG@10 does not follow.** Two-stage 0.0644 vs most-popular 0.0649 vs retrieval 0.0516. More of the relevant set is in the top 10; the first hit is not clearly better-placed than a popularity list.

**Ceiling.** Retrieval Recall@500 = 0.5267 overall, 0.5381 cold-user, 0.2548 cold-item. On train, 8,367 / 10,000 positives were already in the 500 (hit rate 0.8367); 1,633 were injected so the ranker could see them. A test item that never entered the 500 cannot be ranked.

**Cold user.** Two-stage Recall@10 0.0977 vs most-popular 0.0774 vs mean-embedding retrieval 0.0681. The ranker is using item counts and the retrieval score for users with no ID embedding, not a user content tower.

**Cold item.** Two-stage Recall@10 0.0110 vs retrieval 0.0048 vs most-popular 0. At K=100 the ranker is worse than retrieval (0.0376 vs 0.0405). Popularity features bury some cold games that the content tower had placed in the 500.

**Importances** (`results/gbdt_ranker_importance.csv`): `item_n` > `retrieval_score` > `item_n_7d` > `item_days_since_release` > windowed item counts. Weakest: `item_is_free`, `ui_n`, `item_log_price`. No zero-gain feature. `ui_n` is nearly useless for the same last-review-per-pair reason as Phase 2. `retrieval_score` is the two-stage glue — without it the ranker would be a popularity reranker of a popularity-ish candidate set.

**Deciles.** Two-stage Recall@10 is non-zero on deciles 1–2 (0.0016, 0.0022) where most-popular and the two-tower are 0. Decile 10 is 0.1022 vs most-popular 0.0837 vs two-tower 0.0724. Still a head model.

**Not a random-split number.** Ranking was not run on the random split. PIT is undefined in the same way there: test events can precede train events.

---

## Phase 4 — Experimentation layer

Run `20260914T212349Z`. Source: `results/phase4_ope_20260914T212349Z.json`. **No online experiment was run.**

**Setup.** 10,000 temporal warm test users. Candidate set = frozen two-tower top 500 (exact IP). Logging policy = rank-softmax of that list (τ=1). Target = rank-softmax of the PIT GBDT order of the same 500. One simulated impression per user. Reward = 1 iff the sampled item is a held-out purchase. IPS clipped at 20. Interleaving = team-draft @10; clicks = held-out purchases in the slate. MDE DAU values are hypothetical.

**IPS/SNIPS.** Self-check works: IPS(π0 | π0 logs) = 0.0587 = MC of the logger. Target IPS = 0.0083 (analytic 95% CI [0.0039, 0.0128], bootstrap [0.0048, 0.0135]) against on-policy MC 0.0619. ESS 191 / 10,000. Mean weight 0.104; max unclipped weight 1,097. Clip fraction 0.001. SNIPS = 0.0799 with bootstrap 95% CI [0.0482, 0.1289] — wide enough to contain both MC values. Rank-softmax logs are too peaked for IPS to estimate a reranker. That is the result, not a bug in the arithmetic.

**Interleaving.** Most-popular (A) vs GBDT (B): 1,669 / 1,118 / 7,213 (A wins / B wins / ties). P(A wins | decisive) = 0.599, z=10.44. Two-tower vs GBDT: 1,210 / 1,263 / 7,527, p=0.29, a coin flip. Same users, GBDT Recall@10 0.0786 vs most-popular 0.0603 and hit@10 0.2923 vs 0.2630. Most-popular P@1 is 0.1041 vs GBDT 0.0719. Team-draft pays the team that contributed the clicked item; popularity owns the first slot.

**MDE.** Metric treated as Bernoulli hit@10, control p=0.263. α=0.05, power=0.8, 50/50, one session per user per day. A 1% relative lift needs 441,303 users per arm (89 days at a hypothetical 10k DAU, 9 days at 100k, 1 day at 1M). The observed 11.1% relative hit@10 lift (GBDT vs pop on this sample) needs 3,664 per arm — one day at 10k DAU *if it transferred*. IPS did not recover it, so the offline lift is not a green light.

**What this cannot claim.** No production logs, no position bias from real UI, no live feedback loop, no actual A/B test.

---

## Phase 5 — Serving

**Not production traffic.** Both stages behind one FastAPI `POST /recommend`: frozen content two-tower retrieves 500 (exact inner product), PIT GBDT reranks. Same `PITState.features` as training. Terraform reused the `llm-serving-bench` pattern (budget first, public subnet, no NAT) with CPU `t3.medium` instead of GPU. No GitHub OIDC provider was created (the account already has one).

### Skew

`results/skew_check.json`, run `20260915T002631Z`. 200 random (user, item) pairs. Serving `feature_frame` vs `PITState.features` at the temporal cutoff: **0 mismatches**. `retrieval_score` was compared at 0 in this check; `recommend()` fills it from the frozen two-tower.

### Latency

Source: `results/serving_latency.csv`. 200 requests, concurrency 8, k=20, errors=0.

| hardware | p50 ms | p95 ms | p99 ms | run |
|---|---|---|---|---|
| **aws_t3.medium (headline)** | 194.7 | 275.3 | 302.7 | `20260915T004053Z` |
| local_cpu (comparison) | 159.7 | 218.9 | 290.1 | `20260915T002828Z` |

AWS numbers are end-to-end through the public ALB. Local is the same app on the laptop and is labeled as such.

### Infra

Budget `steam-recsys-monthly` ($8 alarm / $10 ceiling) was applied before VPC or compute. ASG started at desired=0, scaled to 1 for the timed session, then `terraform destroy` removed **27 resources**. Post-destroy console check in us-west-2: no `steam-recsys` EC2, no ALB, no NAT. No NAT Gateway existed. Spend is not logged from Cost Explorer in this session; the live window was minutes of `t3.medium` + ALB.

### What this cannot claim

No production traffic, no real users, no autoscaling under real load, no SLA.

---

## Phase 6 — Write-up and Azure port

README follows PROJECT_BRIEF §14. Azure Terraform in `terraform/azure` is a port of `terraform/aws`: resource-group budget first, public VNet only, no NAT Gateway, ACR + blob instead of ECR + S3, Standard Load Balancer instead of ALB, CPU B-series, `instance_count` 0/1. Same $8 / $10 ceiling.

Live apply ran on subscription `Azure subscription 1` after the account existed. Budget `steam-recsys-monthly` ($8) went on `steam-recsys-rg` before VNet or compute. ACR `steamrecsys9086e4` and storage `recs9086e4` stayed in `eastus`. `Standard_B2s`, `Standard_B2ms`, and zonal `Standard_B2s_v2` in `eastus` returned `SkuNotAvailable`. `Standard_B2s` in `westus2` also returned `SkuNotAvailable`. Compute moved to `westus2` (`compute_location`) and `Standard_B2s_v2` created. Azure Linux VMs rejected the ed25519 key; an RSA key was required. NSG probe priority 90 was invalid (Azure requires 100–4096).

Loadgen `20260916T085546Z` against the Standard LB (`POST /recommend`, retrieve 500, PIT GBDT, k=20, concurrency 8, 200 requests, errors=0). Hardware label `azure_Standard_B2s_v2`. Source: `results/serving_latency.csv`.

| hardware | p50 ms | p95 ms | p99 ms | run |
|---|---|---|---|---|
| **aws_t3.medium (headline)** | 194.7 | 275.3 | 302.7 | `20260915T004053Z` |
| azure_Standard_B2s_v2 (port) | 129.8 | 174.5 | 228.2 | `20260916T085546Z` |
| local_cpu (comparison) | 159.7 | 218.9 | 290.1 | `20260915T002828Z` |

Azure numbers are one timed session through a public Standard LB. They are not production traffic and they do not replace the AWS headline. No Cost Management extract was pulled. No NAT Gateway was created.

`terraform destroy` removed **22 resources**, including the resource group. Post-destroy CLI check on the same subscription: 0 `steam-recsys` resource groups, 0 VMs, 0 load balancers, 0 NAT gateways, 0 public IPs, 0 ACR, 0 `recs*` storage accounts.

### What this cannot claim

No Azure production recsys. No second model. No GPU. No overnight serving.

---
