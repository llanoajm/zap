"""Shared utilities for the Exxon/Venezuela-Guyana analysis pipeline.

Design goals (per the brief: robust, reliable, resilient):
  * Zero third-party dependencies -- standard library only. Nothing to install,
    nothing to break across environments or python versions.
  * Every network call retries with exponential backoff and falls back to an
    on-disk cache, so the pipeline always returns *an* answer even when a data
    source is rate-limited (GDELT throttles to 1 req / 5s) or unreachable.
  * All state is on disk (JSON), so the loop can be killed at any point -- by a
    crash, a usage limit, or a container reclaim -- and resume where it left off.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE_DIR = os.path.join(ROOT, "artifacts", "cache")
ARTIFACT_DIR = os.path.join(ROOT, "artifacts")
STATE_PATH = os.path.join(ROOT, "state.json")
CONFIG_PATH = os.path.join(ROOT, "config.json")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

USER_AGENT = "exxon-venezuela-research/1.0 (open-data; stdlib)"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path, default=None):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)  # atomic; survives interruption mid-write


def load_config():
    cfg = load_json(CONFIG_PATH)
    if cfg is None:
        raise SystemExit("config.json missing or invalid")
    return cfg


def _cache_path(key: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in key)[:120]
    return os.path.join(CACHE_DIR, safe + ".cache")


def http_get(url, *, max_retries=4, base_delay=5.0, timeout=30, cache_key=None,
             cache_ttl_sec=3600, throttle_sec=0.0):
    """GET a URL as text. Retries with exponential backoff; on total failure or
    a fresh-enough cache, returns the cached body. Returns (text, from_cache).

    Returns (None, False) only if there is no cache and every attempt failed.
    """
    key = cache_key or url
    cp = _cache_path(key)

    # Serve fresh cache without hitting the network (politeness + resilience).
    cached = load_json(cp)
    if cached and (time.time() - cached.get("ts", 0)) < cache_ttl_sec:
        return cached["body"], True

    if throttle_sec:
        time.sleep(throttle_sec)

    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            save_json(cp, {"ts": time.time(), "url": url, "body": body})
            return body, False
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_err = e
            code = getattr(e, "code", None)
            # 429 / transient: back off harder.
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
        except Exception as e:  # noqa: BLE001 - never let a fetch kill the loop
            last_err = e
            time.sleep(base_delay * (2 ** attempt))

    # Network exhausted: fall back to stale cache if we have one.
    if cached:
        return cached["body"], True
    print(f"[http_get] FAILED {url}: {last_err}")
    return None, False


def http_get_json(url, **kw):
    body, from_cache = http_get(url, **kw)
    if body is None:
        return None, from_cache
    try:
        return json.loads(body), from_cache
    except json.JSONDecodeError:
        return None, from_cache
