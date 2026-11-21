from datetime import date, timedelta

import pytest

from dvx.scan import calibration
from dvx.scan.calibration import CalibrationUnavailable, fetch_calibration_texts


class FakeWiki:
    """Canned MediaWiki: 30 titles per start date, 12 per logevents page."""

    def __init__(self, short_every=0, skew_days=0, bad_titles=False):
        self.skew_days = skew_days
        self.bad_titles = bad_titles
        self.calls = []
        self.short_every = short_every

    def _title(self, start, i):
        if self.bad_titles and i % 3 == 0:
            return f"List of T{start}-{i}"
        if self.bad_titles and i % 3 == 1:
            return f"T{start}-{i} (disambiguation)"
        return f"T{start}-{i}"

    def _ts(self, start):
        d = date.fromisoformat(start) + timedelta(days=self.skew_days)
        return d.isoformat() + "T00:00:00Z"

    def __call__(self, params):
        self.calls.append(dict(params))
        if params.get("list") == "logevents":
            start = params["lestart"][:10]
            first = int(params.get("lecontinue", 0))
            events = [
                {"title": self._title(start, i), "timestamp": self._ts(start)}
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
            extract = "x" * length
            if self.bad_titles and t.endswith("-2"):
                extract = "Foo may refer to:" + "x" * 1500
            pages[str(n)] = {"title": t, "extract": extract}
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
    assert (tmp_path / "c" / "wiki_v2_2025-01-01_16.json").exists()
    again = fetch_calibration_texts(
        date(2025, 1, 1), 16, tmp_path / "c", fetch_json=fake, today=today
    )
    assert again == (seen, unseen)
    assert len(fake.calls) == n


def test_params_and_future_dates_skipped(tmp_path):
    fake = FakeWiki()
    # unseen_after + 30/60/90/120 are past, +180 and later are in the future
    fetch_calibration_texts(
        date(2026, 6, 1), 8, tmp_path, fetch_json=fake, today=date(2026, 10, 30)
    )
    logs = [c for c in fake.calls if c.get("list") == "logevents"]
    starts = sorted({c["lestart"][:10] for c in logs})
    assert starts == [
        "2018-07-15", "2018-10-01", "2019-01-01", "2019-04-01",
        "2019-07-01", "2019-10-01", "2020-01-01", "2020-04-01",
        "2026-07-01", "2026-07-31", "2026-08-30", "2026-09-29",
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


def test_events_far_from_requested_start_dropped(tmp_path):
    fake = FakeWiki(skew_days=400)  # log "collapsed": events a year after lestart
    with pytest.raises(CalibrationUnavailable):
        fetch_calibration_texts(
            date(2025, 1, 1), 16, tmp_path, fetch_json=fake, today=date(2026, 10, 30)
        )
    assert not any(c.get("list") != "logevents" for c in fake.calls)


def test_disambiguation_and_list_pages_excluded(tmp_path):
    fake = FakeWiki(bad_titles=True)
    seen, unseen = fetch_calibration_texts(
        date(2025, 1, 1), 16, tmp_path, fetch_json=fake, today=date(2026, 10, 30)
    )
    assert len(seen) == 16 and len(unseen) == 16
    requested = [
        t for c in fake.calls if c.get("prop") == "extracts" for t in c["titles"].split("|")
    ]
    assert requested  # bad ones were fetched but must not be returned
    assert not any("may refer to" in t for t in seen + unseen)
