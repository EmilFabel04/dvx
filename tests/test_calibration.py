from datetime import date

import pytest

from dvx.scan import calibration
from dvx.scan.calibration import CalibrationUnavailable, fetch_calibration_texts


class FakeWiki:
    """Canned MediaWiki: 30 titles per start date, 12 per logevents page."""

    def __init__(self, short_every=0):
        self.calls = []
        self.short_every = short_every

    def __call__(self, params):
        self.calls.append(dict(params))
        if params.get("list") == "logevents":
            start = params["lestart"][:10]
            first = int(params.get("lecontinue", 0))
            events = [
                {"title": f"T{start}-{i}", "timestamp": start}
                for i in range(first, min(first + 12, 30))
            ]
            out = {"query": {"logevents": events}}
            if first + 12 < 30:
                out["continue"] = {"lecontinue": str(first + 12), "continue": "-||"}
            return out
        assert params["prop"] == "extracts"
        titles = params["titles"].split("|")
        assert len(titles) <= 20
        pages = {}
        for n, t in enumerate(titles):
            length = 100 if self.short_every and n % self.short_every == 0 else 1500
            pages[str(n)] = {"title": t, "extract": "x" * length}
        return {"query": {"pages": pages}}


def test_fetch_uses_cache(tmp_path):
    fake = FakeWiki()
    today = date(2026, 10, 30)
    seen, unseen = fetch_calibration_texts(
        date(2025, 1, 1), 16, tmp_path / "c", fetch_json=fake, today=today
    )
    assert len(seen) == 16 and len(unseen) == 16
    assert all(400 <= len(t) <= 1000 for t in seen + unseen)
    n = len(fake.calls)
    assert n > 0
    assert (tmp_path / "c" / "wiki_2025-01-01_16.json").exists()
    again = fetch_calibration_texts(
        date(2025, 1, 1), 16, tmp_path / "c", fetch_json=fake, today=today
    )
    assert again == (seen, unseen)
    assert len(fake.calls) == n


def test_params_and_future_dates_skipped(tmp_path):
    fake = FakeWiki()
    # unseen_after + 30/90 are past, +180/270 are in the future
    fetch_calibration_texts(
        date(2026, 6, 1), 8, tmp_path, fetch_json=fake, today=date(2026, 10, 30)
    )
    logs = [c for c in fake.calls if c.get("list") == "logevents"]
    starts = sorted({c["lestart"][:10] for c in logs})
    assert starts == [
        "2016-03-01", "2017-03-01", "2018-03-01", "2019-03-01",
        "2026-07-01", "2026-08-30",
    ]
    c = logs[0]
    assert c["letype"] == "create" and c["lenamespace"] == 0
    assert c["ledir"] == "newer" and c["lelimit"] == 500


def test_short_intros_filtered(tmp_path):
    fake = FakeWiki(short_every=2)
    seen, unseen = fetch_calibration_texts(
        date(2025, 1, 1), 16, tmp_path, fetch_json=fake, today=date(2026, 10, 30)
    )
    assert len(seen) == 16 and all(len(t) >= 400 for t in seen + unseen)


def test_too_few_texts_raises(tmp_path):
    def empty(params):
        return {"query": {"logevents": []}}

    with pytest.raises(CalibrationUnavailable):
        fetch_calibration_texts(date(2025, 1, 1), 16, tmp_path, fetch_json=empty)


def test_fetch_failure(tmp_path):
    def boom(params):
        raise OSError("offline")

    with pytest.raises(CalibrationUnavailable):
        fetch_calibration_texts(date(2025, 1, 1), 16, tmp_path, fetch_json=boom)


def test_wikipedia_get_wraps_errors(monkeypatch):
    def boom(req, timeout):
        raise TimeoutError("slow")

    monkeypatch.setattr(calibration.urllib.request, "urlopen", boom)
    with pytest.raises(CalibrationUnavailable):
        calibration.wikipedia_get({"action": "query"})
