"""Edit distance and alignment over arbitrary sequences (code points, letters, words).

Uses rapidfuzz when installed; otherwise a pure-Python implementation that is exact
and fast enough for the benchmark's line- and leaf-sized texts.
"""

from __future__ import annotations

from collections.abc import Sequence

try:  # pragma: no cover - exercised implicitly when the extra is installed
    from rapidfuzz.distance import Levenshtein as _RF
except Exception:  # noqa: BLE001
    _RF = None


def levenshtein(a: Sequence, b: Sequence) -> int:
    if _RF is not None:
        return _RF.distance(a, b)
    return _levenshtein_py(a, b)


def _levenshtein_py(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def align(ref: Sequence, hyp: Sequence) -> list[tuple[str, object, object]]:
    """Minimal edit script as (op, ref_unit, hyp_unit) with op in {=, S, D, I}.
    Used for confusion analysis on the leaderboard."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, up = d[i], d[i - 1]
        for j in range(1, m + 1):
            row[j] = min(up[j] + 1, row[j - 1] + 1, up[j - 1] + (ri != hyp[j - 1]))
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]
