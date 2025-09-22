import hashlib
import json
from pathlib import Path
import os
import time
import requests

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "IAM-Policy-Gen/0.1 (S3-only, service-reference-json)"}

S3_REF_URL = "https://servicereference.us-east-1.amazonaws.com/v1/s3/s3.json"  # AWS official
# Doc: https://docs.aws.amazon.com/service-authorization/latest/reference/service-reference.html

def _cache_path(url: str) -> Path:
    h = hashlib.sha256(url.encode()).hexdigest()[:24]
    return RAW_DIR / f"{h}.json"

def fetch_s3_reference_json(force: bool=False, timeout: int=30) -> dict:
    """Fetches the AWS Service Reference JSON for S3 with a simple file cache."""
    path = _cache_path(S3_REF_URL)
    if path.exists() and not force:
        # Robust read with small retries to tolerate concurrent writes
        attempts = 0
        while True:
            try:
                return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except json.JSONDecodeError:
                attempts += 1
                if attempts >= 5:
                    raise
                time.sleep(0.02)

    r = requests.get(S3_REF_URL, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    # Atomic write: write to temp then rename
    import threading
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}" )
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
    return data
