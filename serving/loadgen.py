"""HTTP load generator for POST /recommend. Adapted from llm-serving-bench/bench/loadgen.py."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.io import utc_now, write_json
from data.paths import RESULTS, SPLITS


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


@dataclass
class LoadReport:
    hardware: str
    concurrency: int
    warmup_n: int
    requests: int
    errors: int
    e2e_p50_ms: float
    e2e_p95_ms: float
    e2e_p99_ms: float
    notes: str = ""
    outcomes_errors: list[str] = field(default_factory=list)


async def _one(
    client: httpx.AsyncClient,
    url: str,
    user_id: str,
    k: int,
    timeout_s: float,
) -> tuple[bool, float, str]:
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json={"user_id": user_id, "k": k}, timeout=timeout_s)
        ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code >= 400:
            return False, ms, f"HTTP {resp.status_code}: {resp.text[:180]}"
        return True, ms, ""
    except Exception as exc:  # noqa: BLE001
        ms = (time.perf_counter() - t0) * 1000.0
        return False, ms, str(exc)


async def wait_ready(base_url: str, timeout_s: float = 600.0) -> None:
    url = base_url.rstrip("/") + "/ready"
    deadline = time.perf_counter() + timeout_s
    async with httpx.AsyncClient() as client:
        while time.perf_counter() < deadline:
            try:
                resp = await client.get(url, timeout=5.0)
                if resp.status_code == 200:
                    return
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(2.0)
    raise TimeoutError(f"/ready did not succeed within {timeout_s}s at {url}")


async def run_load(
    base_url: str,
    user_ids: list[str],
    *,
    concurrency: int = 8,
    warmup_n: int = 5,
    total_requests: int = 200,
    k: int = 20,
    timeout_s: float = 60.0,
) -> LoadReport:
    url = base_url.rstrip("/") + "/recommend"
    async with httpx.AsyncClient() as client:
        for i in range(warmup_n):
            await _one(client, url, user_ids[i % len(user_ids)], k, timeout_s)
        sem = asyncio.Semaphore(concurrency)
        latencies: list[float] = []
        errors: list[str] = []

        async def worker(idx: int) -> None:
            async with sem:
                ok, ms, err = await _one(client, url, user_ids[idx % len(user_ids)], k, timeout_s)
                if ok:
                    latencies.append(ms)
                else:
                    errors.append(err)

        await asyncio.gather(*(worker(i) for i in range(total_requests)))

    return LoadReport(
        hardware="",
        concurrency=concurrency,
        warmup_n=warmup_n,
        requests=total_requests,
        errors=len(errors),
        e2e_p50_ms=percentile(latencies, 50),
        e2e_p95_ms=percentile(latencies, 95),
        e2e_p99_ms=percentile(latencies, 99),
        outcomes_errors=errors[:8],
    )


def sample_user_ids(n: int, seed: int = 42) -> list[str]:
    test = pd.read_parquet(SPLITS / "train_temporal_labeled.parquet")
    users = test["user_id"].astype(str).drop_duplicates().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(users)
    return users[:n]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load-generate POST /recommend.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--hardware", required=True, help="e.g. local_cpu or aws_t3.small")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--n-users", type=int, default=200)
    parser.add_argument("--ready-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    users = sample_user_ids(args.n_users)
    asyncio.run(wait_ready(args.base_url, timeout_s=args.ready_timeout))
    report = asyncio.run(
        run_load(
            args.base_url,
            users,
            concurrency=args.concurrency,
            warmup_n=args.warmup,
            total_requests=args.requests,
            k=args.k,
        )
    )
    report.hardware = args.hardware
    payload = asdict(report)
    payload["run_id"] = utc_now().replace(":", "").replace("-", "")
    payload["phase"] = 5
    payload["base_url"] = args.base_url
    payload["note"] = (
        "Not production traffic. Timed HTTP /recommend against a short-lived endpoint."
    )
    out = RESULTS / f"serving_latency_{payload['run_id']}.json"
    write_json(out, payload)
    print(json.dumps({k: payload[k] for k in ("hardware", "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms", "errors", "requests")}, indent=2))
    print(f"wrote {out}")
    return 0 if report.errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
