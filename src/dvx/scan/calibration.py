"""Calibration texts for the exposure estimate: Wikipedia intros created before
(seen) and after (unseen) a model's training cutoff. See spec section 5.1."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

API_URL = "https://en.wikipedia.org/w/api.php"
CALIBRATION_SOURCE = "English Wikipedia article intros"
USER_AGENT = "dvx/0.1 (https://github.com/EmilFabel04/dvx)"
# The enwiki letype=create log begins on 2018-06-27, so earlier starts collapse onto it.
SEEN_STARTS = [
    date(2018, 7, 15), date(2018, 10, 1), date(2019, 1, 1), date(2019, 4, 1),
    date(2019, 7, 1), date(2019, 10, 1), date(2020, 1, 1), date(2020, 4, 1),
]
UNSEEN_OFFSETS_DAYS = (30, 60, 90, 120, 180, 240, 300, 360)
MAX_EVENT_LAG = timedelta(days=30)
MIN_CHARS = 400
MAX_CHARS = 1000
TITLES_PER_CALL = 20
MAX_PAGES_PER_START = 10


class CalibrationUnavailable(RuntimeError):
    """Calibration texts could not be obtained."""


@dataclass(frozen=True)
class CalibrationSet:
    """Seen and unseen texts, and the date (ISO) they were fetched; None if unrecorded."""

    seen: list[str]
    unseen: list[str]
    fetched_on: str | None


def calibration_sha256(seen: list[str], unseen: list[str]) -> str:
    """SHA-256 of the texts as compact JSON {"seen": [...], "unseen": [...]} (UTF-8)."""
    blob = json.dumps(
        {"seen": seen, "unseen": unseen}, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def wikipedia_get(params: dict) -> dict:
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise CalibrationUnavailable(f"Wikipedia request failed: {e}") from e


def _intros(titles: list[str], fetch_json: Callable[[dict], dict]) -> list[str]:
    data = fetch_json({
        "action": "query", "format": "json", "prop": "extracts", "exintro": 1,
        "explaintext": 1, "titles": "|".join(titles),
    })
    pages = (data.get("query") or {}).get("pages") or {}
    by_title = {p.get("title"): p.get("extract") or "" for p in pages.values()}
    out = []
    for t in titles:  # keep the requested order for determinism
        text = by_title.get(t, "").strip()
        if "(disambiguation)" in t or t.startswith("List of") or "may refer to" in text.lower():
            continue
        if len(text) >= MIN_CHARS:
            out.append(text[:MAX_CHARS])
    return out


def _near(timestamp: str | None, start: date) -> bool:
    """False for events far after the requested start (the log did not reach it)."""
    try:
        ts = datetime.fromisoformat(timestamp).date()
    except (AttributeError, ValueError):
        return False
    return ts <= start + MAX_EVENT_LAG


def _texts_from(start: date, quota: int, fetch_json, used: set[str]) -> list[str]:
    texts: list[str] = []
    cont: str | None = None
    for _ in range(MAX_PAGES_PER_START):
        params = {
            "action": "query", "format": "json", "list": "logevents", "letype": "create",
            "lenamespace": 0, "ledir": "newer", "lelimit": 500,
            "lestart": f"{start.isoformat()}T00:00:00Z",
        }
        if cont is not None:
            params["lecontinue"] = cont
        data = fetch_json(params)
        events = (data.get("query") or {}).get("logevents", [])
        titles = [e["title"] for e in events if "title" in e and _near(e.get("timestamp"), start)]
        titles = [t for t in dict.fromkeys(titles) if t not in used]
        for i in range(0, len(titles), TITLES_PER_CALL):
            batch = titles[i:i + TITLES_PER_CALL]
            used.update(batch)
            texts.extend(_intros(batch, fetch_json))
            if len(texts) >= quota:
                return texts[:quota]
        cont = (data.get("continue") or {}).get("lecontinue")
        if cont is None:
            break
    return texts


def fetch_calibration_texts(
    unseen_after: date,
    n_per_side: int,
    cache_dir: Path,
    fetch_json: Callable[[dict], dict] = wikipedia_get,
    *,
    today: date | None = None,
) -> CalibrationSet:
    """Seen and unseen Wikipedia intros, cached with their fetch date; raises
    CalibrationUnavailable on failure."""
    cache_dir = Path(cache_dir)
    cache = cache_dir / f"wiki_v2_{unseen_after.isoformat()}_{n_per_side}.json"
    if cache.exists():
        try:
            blob = json.loads(cache.read_text(encoding="utf-8"))
            return CalibrationSet(list(blob["seen"]), list(blob["unseen"]), blob.get("fetched_on"))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass  # unreadable cache: refetch
    today = today or date.today()
    unseen_starts = [unseen_after + timedelta(days=d) for d in UNSEEN_OFFSETS_DAYS]
    unseen_starts = [d for d in unseen_starts if d <= today]
    sides = []
    try:
        for starts in (SEEN_STARTS, unseen_starts):
            used: set[str] = set()
            texts: list[str] = []
            for s in starts:
                texts.extend(_texts_from(s, n_per_side // len(starts), fetch_json, used))
            sides.append(texts)
    except (OSError, ValueError) as e:
        raise CalibrationUnavailable(f"Wikipedia request failed: {e}") from e
    seen, unseen = sides
    need = n_per_side // 2
    if len(seen) < need or len(unseen) < need:
        raise CalibrationUnavailable(
            f"too few calibration texts (seen {len(seen)}, unseen {len(unseen)}, need {need})"
        )
    fetched = CalibrationSet(seen, unseen, fetched_on=today.isoformat())
    cache_dir.mkdir(parents=True, exist_ok=True)
    blob = {"seen": seen, "unseen": unseen, "fetched_on": fetched.fetched_on}
    cache.write_text(json.dumps(blob), encoding="utf-8")
    return fetched
