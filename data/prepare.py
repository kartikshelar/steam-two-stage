from __future__ import annotations

import argparse
import ast
import gzip
import json
import math
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from data.config import load_config
from data.io import sha256_file, utc_now, write_json
from data.paths import PROCESSED, RAW, RESULTS, ensure_dirs

USER_AGENT = "steam-recsys/0.1 (research; USC; +https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data)"
REVIEWS_NAME = "steam_reviews.json.gz"
GAMES_NAME = "steam_games.json.gz"
BUNDLES_NAME = "bundle_data.json.gz"


def parse_mcauley_line(line: str) -> dict[str, Any] | None:
    """McAuley dumps are 'loose JSON': Python literals, sometimes with u'' prefixes."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    for candidate in (line, line.replace("u'", "'").replace('u"', '"')):
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, dict):
                return obj
        except (ValueError, SyntaxError, MemoryError):
            continue
    # Last resort matches the lab's published parser (`eval(l)`). Source is the
    # McAuley academic dump, not untrusted input.
    obj = eval(line, {"__builtins__": {}}, {})  # noqa: S307
    if isinstance(obj, dict):
        return obj
    return None


def parse_price(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower().replace("$", "")
    if text in {"", "none", "null"}:
        return None
    if "free" in text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def parse_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None and str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def download_with_fallback(urls: list[str], dest: Path) -> tuple[Path, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest, "exists"
    errors: list[str] = []
    for url in urls:
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                progress = tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    desc=f"download {dest.name}",
                )
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    progress.update(len(chunk))
                progress.close()
            tmp.replace(dest)
            return dest, url
        except Exception as exc:  # noqa: BLE001 — try the next mirror
            errors.append(f"{url}: {exc}")
            if tmp.exists():
                tmp.unlink()
    raise RuntimeError("All download URLs failed:\n" + "\n".join(errors))


def iter_gzip_dicts(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    n = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            obj = parse_mcauley_line(line)
            if obj is None:
                continue
            yield obj
            n += 1
            if limit is not None and n >= limit:
                return


def _review_record(obj: dict[str, Any]) -> dict[str, Any] | None:
    user = obj.get("username") or obj.get("user_id") or obj.get("user")
    item = obj.get("product_id") or obj.get("item_id") or obj.get("app_id")
    date = obj.get("date")
    if user is None or item is None or date is None:
        return None
    hours = obj.get("hours")
    try:
        hours_f = float(hours) if hours is not None else float("nan")
    except (TypeError, ValueError):
        hours_f = float("nan")
    return {
        "user_id": str(user),
        "item_id": str(item),
        "hours": hours_f,
        "date_raw": str(date),
        "n_products": obj.get("products"),
        "early_access": bool(obj.get("early_access", False)),
    }


def reviews_to_frame(rows: Iterable[dict[str, Any]], batch_size: int = 200_000) -> pd.DataFrame:
    parts_dir = PROCESSED / "_review_parts"
    if parts_dir.exists():
        for leftover in parts_dir.glob("*.parquet"):
            leftover.unlink()
    else:
        parts_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    part_idx = 0
    batch: list[dict[str, Any]] = []
    part_paths: list[Path] = []

    def flush() -> None:
        nonlocal part_idx, batch
        if not batch:
            return
        path = parts_dir / f"part_{part_idx:04d}.parquet"
        pd.DataFrame.from_records(batch).to_parquet(path, index=False)
        part_paths.append(path)
        part_idx += 1
        batch = []

    for obj in tqdm(rows, desc="parse reviews", unit="row"):
        rec = _review_record(obj)
        if rec is None:
            skipped += 1
            continue
        batch.append(rec)
        if len(batch) >= batch_size:
            flush()
    flush()
    if not part_paths:
        raise RuntimeError("No reviews parsed. Check the dump format.")

    df = pd.concat([pd.read_parquet(p) for p in part_paths], ignore_index=True)
    for p in part_paths:
        p.unlink()
    parts_dir.rmdir()

    df["timestamp"] = pd.to_datetime(df["date_raw"], errors="coerce", utc=True)
    n_bad_date = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"])
    df["ts"] = (df["timestamp"].astype("int64") // 10**9).astype("int64")
    df["hours"] = pd.to_numeric(df["hours"], errors="coerce")
    df = df.sort_values("timestamp")
    n_dupes = int(df.duplicated(["user_id", "item_id"]).sum())
    df = df.drop_duplicates(["user_id", "item_id"], keep="last")
    df.attrs["skipped_missing"] = skipped
    df.attrs["dropped_bad_date"] = n_bad_date
    df.attrs["dropped_duplicate_pairs"] = n_dupes
    return df.reset_index(drop=True)


def games_to_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for obj in rows:
        item_id = obj.get("id") or obj.get("app_id") or obj.get("item_id")
        if item_id is None:
            url = str(obj.get("url") or obj.get("reviews_url") or "")
            if "/app/" in url:
                item_id = url.split("/app/")[1].split("/")[0]
        if item_id is None:
            continue
        records.append(
            {
                "item_id": str(item_id),
                "title": obj.get("title") or obj.get("app_name") or "",
                "genres": json.dumps(parse_str_list(obj.get("genres")), ensure_ascii=False),
                "tags": json.dumps(parse_str_list(obj.get("tags")), ensure_ascii=False),
                "specs": json.dumps(parse_str_list(obj.get("specs")), ensure_ascii=False),
                "price": parse_price(obj.get("price")),
                "release_date_raw": str(obj.get("release_date") or ""),
                "publisher": obj.get("publisher") or "",
                "developer": obj.get("developer") or "",
                "early_access": bool(obj.get("early_access", False)),
                "sentiment": obj.get("sentiment") or "",
                "metascore": pd.to_numeric(obj.get("metascore"), errors="coerce"),
            }
        )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df = df.drop_duplicates("item_id", keep="last")
    df["release_date"] = pd.to_datetime(df["release_date_raw"], errors="coerce", utc=True)
    return df.reset_index(drop=True)


def profile_reviews(reviews: pd.DataFrame, games: pd.DataFrame) -> dict[str, Any]:
    hours = reviews["hours"].dropna()
    user_counts = reviews.groupby("user_id").size()
    item_counts = reviews.groupby("item_id").size()
    review_items = set(reviews["item_id"].astype(str))
    game_items = set(games["item_id"].astype(str)) if not games.empty else set()
    return {
        "n_reviews": int(len(reviews)),
        "n_users": int(reviews["user_id"].nunique()),
        "n_items_in_reviews": int(reviews["item_id"].nunique()),
        "n_items_in_games_metadata": int(games["item_id"].nunique()) if not games.empty else 0,
        "n_items_reviews_not_in_metadata": int(len(review_items - game_items)),
        "n_items_metadata_not_in_reviews": int(len(game_items - review_items)),
        "timestamp_min": str(reviews["timestamp"].min()),
        "timestamp_max": str(reviews["timestamp"].max()),
        "hours_missing": int(reviews["hours"].isna().sum()),
        "hours_min": float(hours.min()) if len(hours) else None,
        "hours_p50": float(hours.quantile(0.50)) if len(hours) else None,
        "hours_p90": float(hours.quantile(0.90)) if len(hours) else None,
        "hours_p99": float(hours.quantile(0.99)) if len(hours) else None,
        "hours_max": float(hours.max()) if len(hours) else None,
        "frac_hours_lt_1": float((hours < 1).mean()) if len(hours) else None,
        "frac_hours_lt_2": float((hours < 2).mean()) if len(hours) else None,
        "frac_hours_lt_10": float((hours < 10).mean()) if len(hours) else None,
        "reviews_per_user_p50": float(user_counts.quantile(0.50)),
        "reviews_per_user_p90": float(user_counts.quantile(0.90)),
        "reviews_per_user_max": int(user_counts.max()),
        "reviews_per_item_p50": float(item_counts.quantile(0.50)),
        "reviews_per_item_p90": float(item_counts.quantile(0.90)),
        "reviews_per_item_max": int(item_counts.max()),
        "frac_users_lt_5_reviews": float((user_counts < 5).mean()),
        "frac_items_lt_5_reviews": float((item_counts < 5).mean()),
        "skipped_missing": int(reviews.attrs.get("skipped_missing", 0)),
        "dropped_bad_date": int(reviews.attrs.get("dropped_bad_date", 0)),
        "dropped_duplicate_pairs": int(reviews.attrs.get("dropped_duplicate_pairs", 0)),
        "source_notes": {
            "mcauley_page_reviews": 7_793_069,
            "mcauley_page_users": 2_567_538,
            "mcauley_page_items": 15_474,
            "sasrec_readme_games": 32_135,
            "license": "No OSI license published. McAuley lab states the dumps are for research and must be cited.",
            "parser": "json.loads, then ast.literal_eval, then restricted eval (lab-published format).",
        },
    }


def popularity_table(reviews: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    counts = reviews.groupby("item_id").size().rename("n_reviews").reset_index()
    hours = reviews.groupby("item_id")["hours"].median().rename("median_hours")
    out = counts.merge(hours, on="item_id", how="left")
    out = out.sort_values("n_reviews", ascending=False).head(top_n)
    return out.reset_index(drop=True)


def prepare(config: dict[str, Any], max_reviews: int | None = None, reuse: bool = True) -> dict[str, Any]:
    ensure_dirs()
    urls = config["data"]["urls"]
    if max_reviews is None:
        max_reviews = config["data"].get("max_reviews")

    reviews_path, reviews_src = download_with_fallback(urls["reviews"], RAW / REVIEWS_NAME)
    games_path, games_src = download_with_fallback(urls["games"], RAW / GAMES_NAME)
    bundles_path, bundles_src = download_with_fallback(urls["bundles"], RAW / BUNDLES_NAME)

    games = games_to_frame(iter_gzip_dicts(games_path))
    reviews_out = PROCESSED / "reviews.parquet"
    games_out = PROCESSED / "games.parquet"
    if reuse and reviews_out.exists() and max_reviews is None:
        print(f"reusing {reviews_out}", flush=True)
        reviews = pd.read_parquet(reviews_out)
        reviews.attrs["skipped_missing"] = 0
        reviews.attrs["dropped_bad_date"] = 0
        reviews.attrs["dropped_duplicate_pairs"] = 0
    else:
        reviews = reviews_to_frame(iter_gzip_dicts(reviews_path, limit=max_reviews))
        reviews.to_parquet(reviews_out, index=False)
    games.to_parquet(games_out, index=False)

    profile = profile_reviews(reviews, games)
    profile.update(
        {
            "created_at": utc_now(),
            "max_reviews_arg": max_reviews,
            "reviews_path": str(reviews_path.relative_to(ROOT)),
            "games_path": str(games_path.relative_to(ROOT)),
            "bundles_path": str(bundles_path.relative_to(ROOT)),
            "reviews_source": reviews_src,
            "games_source": games_src,
            "bundles_source": bundles_src,
            "reviews_sha256": sha256_file(reviews_path),
            "games_sha256": sha256_file(games_path),
            "bundles_sha256": sha256_file(bundles_path),
            "reviews_bytes": reviews_path.stat().st_size,
            "games_bytes": games_path.stat().st_size,
        }
    )
    write_json(RESULTS / "dataset_profile.json", profile)
    pd.DataFrame([{k: v for k, v in profile.items() if not isinstance(v, dict)}]).to_csv(
        RESULTS / "dataset_profile.csv", index=False
    )
    popularity_table(reviews).to_csv(RESULTS / "item_popularity_head.csv", index=False)

    hours = reviews["hours"].dropna()
    if len(hours):
        hist = pd.cut(
            hours.clip(upper=hours.quantile(0.99)),
            bins=[0, 0.5, 1, 2, 5, 10, 20, 50, 100, float("inf")],
            include_lowest=True,
        ).value_counts().sort_index()
        hist.rename_axis("hours_bin").reset_index(name="n").to_csv(
            RESULTS / "hours_histogram.csv", index=False
        )
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and profile the Steam reviews dump.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-reviews", type=int, default=None)
    parser.add_argument("--force-parse", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    profile = prepare(config, max_reviews=args.max_reviews, reuse=not args.force_parse)
    print(json.dumps({k: profile[k] for k in ("n_reviews", "n_users", "n_items_in_reviews", "n_items_in_games_metadata", "timestamp_min", "timestamp_max")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
