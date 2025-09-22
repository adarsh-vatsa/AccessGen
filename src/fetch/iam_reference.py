import hashlib
import json
from pathlib import Path
import requests

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "IAM-Policy-Gen/0.1 (IAM, service-reference-json)"}

IAM_REF_URL = "https://servicereference.us-east-1.amazonaws.com/v1/iam/iam.json"

def _cache_path(url: str) -> Path:
    h = hashlib.sha256(url.encode()).hexdigest()[:24]
    return RAW_DIR / f"{h}.json"

def fetch_iam_reference_json(force: bool=False, timeout: int=30) -> dict:
    """Fetches the AWS Service Reference JSON for IAM with a simple file cache."""
    path = _cache_path(IAM_REF_URL)
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))

    r = requests.get(IAM_REF_URL, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data