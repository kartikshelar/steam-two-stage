from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.paths import RAW, ensure_dirs

USER_AGENT = "steam-recsys/0.1 (research; USC)"

# HuggingFace first: the lab origin is often slow from residential networks.
FILES = {
    "steam_games.json.gz": [
        "https://huggingface.co/datasets/recommender-system/steam-review-and-bundle-dataset/resolve/main/steam_games.json.gz",
        "https://mcauleylab.ucsd.edu/public_datasets/data/steam/steam_games.json.gz",
    ],
    "bundle_data.json.gz": [
        "https://huggingface.co/datasets/recommender-system/steam-review-and-bundle-dataset/resolve/main/bundle_data.json.gz",
        "https://mcauleylab.ucsd.edu/public_datasets/data/steam/bundle_data.json.gz",
    ],
    "steam_reviews.json.gz": [
        "https://huggingface.co/datasets/recommender-system/steam-review-and-bundle-dataset/resolve/main/steam_reviews.json.gz",
        "https://mcauleylab.ucsd.edu/public_datasets/data/steam/steam_reviews.json.gz",
    ],
}


def download(name: str, urls: list[str]) -> Path:
    dest = RAW / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists {name} {dest.stat().st_size}", flush=True)
        return dest
    last_err: Exception | None = None
    for url in urls:
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            print(f"downloading {name} from {url}", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
                n = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    n += len(chunk)
                    if n % (20 << 20) < (1 << 20):
                        print(f"  {name}: {n / 1e6:.1f} MB", flush=True)
            tmp.replace(dest)
            print(f"done {name} {dest.stat().st_size}", flush=True)
            return dest
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"fail {url}: {exc}", flush=True)
            if tmp.exists():
                tmp.unlink()
    raise RuntimeError(f"failed {name}: {last_err}")


def main() -> int:
    ensure_dirs()
    RAW.mkdir(parents=True, exist_ok=True)
    for name, urls in FILES.items():
        download(name, urls)
    print("all files present", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
