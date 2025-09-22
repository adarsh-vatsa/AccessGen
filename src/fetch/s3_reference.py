import hashlib
import json
from pathlib import Path
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
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))

    r = requests.get(S3_REF_URL, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data