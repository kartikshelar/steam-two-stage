# steam-recsys — Project Brief

**Owner:** Kartik Pradip Shelar
**Status:** not started
**Created:** 2026-09-09
**Purpose of this file:** complete, standalone context for building this project.
Anyone (or any agent) reading only this file should be able to start work.

---

## 1. Why this project exists

Kartik is an MS CS student at USC (graduating May 2027) applying to US ML
engineering, ML infrastructure, and SWE roles. His portfolio is strong on
research, evaluation methodology, RAG systems, and (as of the just-completed
`llm-serving-bench`) cloud infrastructure and GPU serving.

Two gaps remain, both flagged in his own interview-prep notes as absent:

1. **Recommender systems.** No experience. This blocks credible application to
   the single largest category of US ML jobs — TikTok, LinkedIn, Meta,
   Pinterest, Snap, Instacart. He has a live TikTok recruiter thread for the
   2027 new-grad cycle.
2. **A/B testing and product analytics.** No experience.

This project closes both. It cannot close the second with a real online
experiment (no users), but it can build the offline machinery that exists
precisely because online experiments are expensive: inverse propensity scoring,
SNIPS, interleaving, minimum detectable effect. That is an honest and
substantive answer to "tell me about your experimentation experience."

**Adjacent experience that does NOT count as recsys:** SourceBound is retrieval
(embeddings, top-k, BM25 hybrid, cross-encoder reranking). Two-tower retrieval
is structurally similar, but recsys-specific problems — implicit feedback, cold
start, popularity bias, feedback loops, ranking on top of retrieval — are not
covered by it. State this distinction honestly.

---

## 2. What "production level" means here

The phrase is vague and vague ambition is how projects die. Concrete definition.
This system is production-grade if and only if it does these three things:

1. **Two stages, not one.** Retrieval narrows the catalog to hundreds; ranking
   orders those hundreds with expensive features. Every real system does this.
   Single-stage projects are notebooks.
2. **Point-in-time correct features.** No feature may use data from after the
   interaction being predicted. This is the most common way real recsys
   pipelines silently break.
3. **An honest answer to "will this work online?"** Offline metrics famously do
   not predict online results. Address it with off-policy estimators and an
   interleaving harness, and state the limits plainly.

Scale is not the criterion. Correctness and honest evaluation are.

---

## 3. Non-negotiable rules

These come from a documented history of resume fabrication errors in this
owner's project history. They are not stylistic preferences.

1. **Every number that leaves this project must be traceable to a logged run.**
   If it is not in `results/`, it does not go in the README and it does not go
   on a resume.
2. **Never report a random-split number as the headline.** Temporal split is the
   real number. Random split appears only as a labeled comparison showing
   inflation.
3. **The popularity baseline is reported prominently, always.** "Recommend the
   most popular items" is embarrassingly strong. If the model barely beats it,
   say so in the README.
4. **Negative results get equal billing.** A feature that doesn't help, a rung
   that fails, a metric that moves the wrong way — all reported.
5. **No claim of production traffic, real users, live feedback loops, or an
   actual A/B test.** None of these exist here. The offline machinery is the
   claim.
6. **This project does not touch `llm-serving-bench` or SourceBound numbers.**
   Separate repos, separate claims.

---

## 4. Dataset

**Steam reviews (McAuley lab, UCSD).**

- ~7.79M reviews, ~2.57M users, ~32,135 games
- Includes **play hours per review** alongside the review text
- Timestamped
- Companion bundle dataset available (games bundled together)
- Mirrored on HuggingFace

Verify current availability, license, and exact schema before building. The
figures above should be re-confirmed from the source, not taken from this file.

### Why Steam and not something else

Rejected alternatives and why:

- **MovieLens** — thousands of identical projects; reads as a tutorial
- **Amazon Reviews** — excellent data, but star ratings make the label problem
  trivial ("4+ is positive"), removing the most interesting part
- **Spotify playlists** — interesting task shape, but no per-user timestamped
  interactions, which kills the point-in-time story
- **Job postings** — no public dataset with real click logs; would require
  fabricating interactions
- **GitHub repos via GHArchive** — genuinely distinctive and the owner already
  knows the GitHub GraphQL API, but building the dataset is a project in itself
  and risks turning this into a data-pipeline project with recsys bolted on

### The property that makes Steam right

**Play time is a genuinely awkward implicit signal, and that awkwardness is the
centerpiece of this project.**

A user bought a game and played 0.3 hours. Another played 2,000 hours. Another
played 40 hours and left a negative review. The signal is neither binary nor
linear. Deciding what "liked it" means is a real modeling decision with no
textbook answer.

Three further properties:

- **Cold start is severe and real.** New releases have zero interactions on
  launch day, exactly when recommending them matters most.
- **Popularity bias is extreme.** A handful of titles dominate. The
  popularity-decile breakdown will show something dramatic.
- **Sequels and franchises create a leakage trap.** Recommending a game's sequel
  to someone who played the original scores well offline and is nearly worthless
  as a recommendation. Detecting and reporting this is a strong finding.

---

## 5. The label experiment (do this first, it is the headline)

Before any model work, decide how play time becomes a preference label. Test at
least three formulations:

1. **Binary purchase** — ignore play time entirely
2. **Thresholded play time** — liked if hours > some cutoff
3. **Normalized against the game's own distribution** — e.g. percentile of play
   time among all players of that title, which corrects for the fact that some
   games are simply longer

Train the same retrieval model on each. Report the metric differences and, more
importantly, **the qualitative difference in what gets recommended.** The
prediction to write down in advance: naive thresholding will over-recommend
long games.

Pre-register that prediction in the README before running, then report whether
it held. This mirrors the pre-registered falsification criteria the owner used
in his research work and is a real differentiator over tutorial projects.

---

## 6. Architecture

```
        user request
             │
             ▼
  ┌─────────────────────┐
  │  STAGE 1: RETRIEVAL │   two-tower, ANN index
  │  32K games → top 500│   target: single-digit ms
  └─────────────────────┘
             │
             ▼
  ┌─────────────────────┐
  │  STAGE 2: RANKING   │   GBDT over rich features
  │  500 → ordered 20   │
  └─────────────────────┘
             │
             ▼
      served endpoint
```

### Stage 1 — Retrieval

- Two-tower model: user tower and item tower, trained to a shared embedding
  space
- **In-batch negatives with logQ correction** for popularity sampling bias.
  Without the correction, popular items are over-penalized as negatives.
- ANN index over item embeddings (FAISS or ScaNN)
- Item tower must accept content features (genre, tags, release date, price) so
  that a game with zero interactions still gets an embedding. This is the cold
  start mechanism.

### Stage 2 — Ranking

- Gradient-boosted trees (LightGBM or XGBoost). This is what most production
  systems actually use.
- Features: user-item interaction counts, recency, trailing-window item
  popularity, user activity level, genre affinity, price, release recency
- Trained on retrieved candidates, not on the full catalog

---

## 7. Point-in-time correctness (the hardest and most valuable part)

Every feature is computed **as of the interaction timestamp**, never later.

- Build a point-in-time join. Write it yourself — roughly 200 lines, and you
  learn more than adopting a feature store.
- Write a test that fails if a future-leaking feature slips in.
- **Then measure the leak.** Train one model with naively-computed (leaked)
  features and one with point-in-time features. Report the gap.

That gap number is the single best interview story in this project. Most
candidates have never measured it.

---

## 8. Evaluation protocol

### Splits

- **Temporal split is the headline.** Train on everything before time T, test
  after.
- **Random split reported alongside, clearly labeled**, to show how much it
  inflates results.

### Metrics

- Recall@K and NDCG@K (K = 10, 50, 100)
- **Broken out by item popularity decile.** Overall numbers hide that most
  models only look good because they recommend popular items.
- **Cold-start users and items reported separately** (under 5 interactions).
- **Most-popular baseline reported prominently.**

### Anti-patterns to avoid

- Reporting only overall Recall@K
- Random split as the headline
- Omitting the popularity baseline because it's embarrassing
- Sampled evaluation against 100 random negatives (inflates everything;
  evaluate against the full catalog)

---

## 9. Experimentation layer (closes the A/B testing gap)

- **IPS and SNIPS estimators** over logged data
- **Interleaving harness** comparing two ranker variants
- **Minimum detectable effect calculation**: given stated traffic assumptions,
  how long would a real online test need to run to detect a given lift
- Variance analysis on the estimators

State plainly in the README: no online experiment was run. This is the offline
machinery used to decide whether an online test is worth running.

---

## 10. Serving

**Reuse the Terraform stack from `llm-serving-bench`.** That is the main reason
this project is on AWS.

- Retrieval + ranking behind a single endpoint
- p50/p95/p99 latency measured under load, using the load generator already
  written for `llm-serving-bench`
- **Training-serving skew check**: assert that features computed in the serving
  path match the training path on a sampled set. Report any mismatch found.

### Cost

This is a cheap project. Training runs on Kaggle free tier. Ranking is CPU.
Serving is a small CPU workload. **Expect under $10 on AWS.**

Same guardrails as `llm-serving-bench`:

- [ ] AWS Budgets alarm set before any Terraform
- [ ] Public subnet only, no NAT Gateway
- [ ] Spot where applicable
- [ ] `terraform destroy` at the end of every session

### Azure port (final phase, optional)

The owner has an Azure student account. **Do not build primarily on Azure** —
that pays a fresh provider setup tax to save a few dollars and spends time on
infrastructure instead of on the recsys machinery, which is the point of the
project.

Instead, once everything works on AWS, port the Terraform to Azure as a
contained final task. One afternoon. This yields a real and useful claim: the
IaC was written once and ported across two providers. Do it as a port, not as
the primary target.

---

## 11. Scope

### In scope

Everything in §5 through §10.

### Explicitly out of scope

Do not add these mid-build. Scope creep is the primary failure mode.

- **Sequence models** (SASRec, BERT4Rec). Interesting, and a different project.
- Real-time streaming features. Batch point-in-time is sufficient to
  demonstrate correctness.
- Multi-objective or multi-task ranking heads.
- Any UI.
- A general-purpose feature store product.
- LLM-based recommendation or text embeddings of reviews. Tempting given the
  owner's background; it is a fourth stage and it is not needed.

---

## 12. Phase plan

Each phase has a definition of done. Do not start the next until the current is
met.

### Phase 0 — Data and labels
**Done when:** dataset downloaded and profiled; temporal split defined and
frozen with a manifest; three label formulations implemented; the label
experiment from §5 run and written up; predictions pre-registered before the
runs.

### Phase 1 — Retrieval
**Done when:** two-tower model trained with logQ-corrected in-batch negatives;
ANN index built; Recall@K and NDCG@K reported on the temporal split, broken out
by popularity decile, with the most-popular baseline alongside; cold-start
performance reported separately.

### Phase 2 — Point-in-time features
**Done when:** point-in-time join implemented; leak-detection test passing; the
leaked-vs-correct model comparison run and the gap reported.

### Phase 3 — Ranking
**Done when:** GBDT ranker trained over retrieved candidates; two-stage
end-to-end metrics reported against the retrieval-only baseline; feature
importances written up, including features that did not help.

### Phase 4 — Experimentation layer
**Done when:** IPS and SNIPS implemented with variance analysis; interleaving
harness working; MDE calculation documented with its traffic assumptions stated.

### Phase 5 — Serving
**Done when:** both stages behind one endpoint on the reused Terraform stack;
latency measured under load; training-serving skew check passing or its
mismatch documented; `terraform destroy` verified clean; spend under $10.

### Phase 6 — Write-up and Azure port
**Done when:** README complete per §14; Terraform ported to Azure and verified.

**Estimated total:** 4–5 weekends. The expensive resource is time, not compute.

---

## 13. Repository structure

```
steam-recsys/
├── README.md
├── data/
│   ├── prepare.py           # download, profile, filter
│   ├── splits.py            # temporal + random, manifest-backed
│   └── labels.py            # the three play-time formulations
├── features/
│   ├── point_in_time.py     # the PIT join
│   └── test_no_leakage.py   # fails if a future-leaking feature appears
├── retrieval/
│   ├── two_tower.py
│   ├── train.py             # in-batch negatives + logQ correction
│   └── index.py             # FAISS/ScaNN build + query
├── ranking/
│   ├── features.py
│   └── train_gbdt.py
├── eval/
│   ├── metrics.py           # Recall@K, NDCG@K, by popularity decile
│   ├── baselines.py         # most-popular, random
│   └── coldstart.py
├── experiments/
│   ├── ips.py               # IPS + SNIPS + variance
│   ├── interleaving.py
│   └── mde.py
├── serving/
│   ├── app.py               # both stages, one endpoint
│   ├── Dockerfile
│   └── skew_check.py
├── terraform/
│   ├── aws/
│   └── azure/               # Phase 6 port
├── results/
│   └── *.csv                # every logged run
└── docs/
    └── findings.md          # per-phase analysis, including what failed
```

---

## 14. README requirements

The README is what a recruiter or interviewer actually reads.

Must contain, in this order:

1. One-line statement of what this is
2. **The label experiment result** — this is the hook, lead with it
3. Two-stage architecture diagram
4. Results table: temporal split headline, random split labeled as the inflated
   comparison, most-popular baseline, popularity-decile breakdown, cold start
5. **The point-in-time leak gap** — leaked vs correct features
6. Serving latency, p50/p95/p99
7. Experimentation section, with the limits stated plainly
8. **A "What didn't work" section.** Give it real space.
9. Explicit statement of what was not built and why: no online experiment, no
   real users, no live feedback loop, no sequence models
10. Reproduction instructions

---

## 15. What this project still cannot claim

Be honest about these in interviews. Overclaiming here is worse than the gap.

- No production recsys experience
- No live feedback loop, so no experience with degenerate loops or position bias
  in a deployed system
- No real A/B test
- No real-time feature serving
- No exposure to the organizational side (metric selection, guardrail metrics,
  launch decisions)

The honest framing: built the offline machinery and can reason about the
problems. Has not operated one in production.

---

## 16. Failure modes to watch for

| Risk | Mitigation |
|---|---|
| Scope creep (sequence models, LLM features, a UI) | §11 out-of-scope list is binding |
| Random split reported as headline | §8; temporal is the headline, always |
| Popularity baseline quietly omitted | §3 rule 3 |
| Sampled negatives inflating eval | Evaluate against the full catalog |
| Point-in-time correctness skipped as "too fiddly" | It is the highest-value part; Phase 2 gate blocks progress without it |
| Model changed mid-project invalidating comparisons | Fix the retrieval architecture at end of Phase 1 |
| Azure port attempted first | §10; AWS first, port last |
| Project displaces applications and DSA | This is a side track. Applications and coding practice remain the priority. |

---

## 17. Context on the owner (for tone and framing)

- Prefers direct output over preamble. Disagreement is welcome and expected.
- Has a documented history of resume claim errors, all traced to the same cause:
  reconstructing a fact from an earlier draft instead of reading the source.
  Always re-read the source.
- Reports negative results honestly as standing practice — a falsified formal
  proposition, a permutation null of p = 0.942 that overturned his own group's
  headline, six measured RAG upgrades that none beat baseline. This project
  follows the same standard.
- Existing strengths this project is not trying to duplicate: evaluation
  methodology, representation learning research, RAG systems, GPU serving and
  cloud infrastructure.
- The gaps being closed are narrow and specific: recommender systems, and
  offline experimentation methodology.
