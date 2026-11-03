from dvx.data import Example
from dvx.scan.duplicates import duplicate_stats, normalize_text


def ex(prompt: str, response: str, line: int = 1) -> Example:
    return Example(prompt=prompt, response=response, system=None, line=line)


def words(prefix: str, n: int = 60) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_normalize_text():
    assert normalize_text("  Hello \n\t World Ａ ") == "hello world a"


def test_exact_after_normalization():
    stats = duplicate_stats([ex("Hello  World", "x"), ex("hello world", "X", 2)])
    assert stats.exact_dup == 1
    assert stats.near_dup == 0


def test_near_duplicate():
    base = words("w")
    changed = base.replace("w30", "zzz")
    stats = duplicate_stats([ex("q", base), ex("q", changed, 2)])
    assert stats.near_dup == 1
    assert stats.exact_dup == 0


def test_distinct():
    stats = duplicate_stats([ex("q", words("a")), ex("q", words("b"), 2)])
    assert (stats.exact_dup, stats.near_dup) == (0, 0)


def test_fractions():
    examples = [ex("a", "1"), ex("b", "2", 2), ex("c", "3", 3), ex("a", "1", 4)]
    stats = duplicate_stats(examples)
    assert stats.n == 4
    assert stats.exact_fraction == 0.25
    assert stats.near_fraction == 0.0


def test_empty():
    stats = duplicate_stats([])
    assert (stats.n, stats.exact_fraction, stats.near_fraction) == (0, 0.0, 0.0)
