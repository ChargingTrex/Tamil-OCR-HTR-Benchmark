"""Binary "unit tests" for multi-line readings, after olmOCR-Bench (Poznanski et al. 2025).

A character error rate over a whole page punishes one misplaced line as heavily as many
misread letters. Pass/fail tests separate the two:

* **present** — each line of the reference occurs somewhere in the answer, with at most
  10 % of its characters changed (approximate substring match);
* **order** — each pair of consecutive reference lines occurs in the same order.

Tests are made only for references with two or more lines of at least ``MIN_CHARS``
characters, after the subset's normalisation. They are reported, never ranked.
"""

from __future__ import annotations

MIN_CHARS = 5
MAX_CHANGE = 0.10


def best_match(pattern: str, text: str) -> tuple[int, int]:
    """(fewest edits that turn ``pattern`` into some substring of ``text``, end index of the
    leftmost such substring) — Myers' bit-parallel approximate string matching."""
    m = len(pattern)
    if m == 0:
        return 0, -1
    peq: dict[str, int] = {}
    for i, ch in enumerate(pattern):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    mask = (1 << m) - 1
    high = 1 << (m - 1)
    pv, mv, score = mask, 0, m
    best, best_end = m, -1
    for j, ch in enumerate(text):
        eq = peq.get(ch, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & high:
            score += 1
        elif mh & high:
            score -= 1
        ph = (ph << 1) & mask
        mh = (mh << 1) & mask
        pv = mh | (~(xv | ph) & mask)
        mv = ph & xv
        if score < best:
            best, best_end = score, j
    return best, best_end


def make_tests(lines: list[str]) -> list[str]:
    """The reference lines that get tests (already normalised)."""
    return [ln for ln in lines if len(ln.replace(" ", "")) >= MIN_CHARS]


def run_tests(lines: list[str], answer: str) -> dict:
    """Run the present and order tests of ``lines`` against a normalised answer."""
    if len(lines) < 2:
        return {"present": 0, "present_n": 0, "order": 0, "order_n": 0}
    ends = []
    for ln in lines:
        d, end = best_match(ln, answer)
        ends.append(end if d <= MAX_CHANGE * len(ln) else None)
    order = sum(1 for a, b in zip(ends, ends[1:]) if a is not None and b is not None and a < b)
    return {"present": sum(e is not None for e in ends), "present_n": len(ends),
            "order": order, "order_n": len(ends) - 1}


def summarize(results: list[dict]) -> dict | None:
    """Pool per-item test results into pass rates."""
    p = sum(r["present"] for r in results)
    pn = sum(r["present_n"] for r in results)
    o = sum(r["order"] for r in results)
    on = sum(r["order_n"] for r in results)
    if not pn:
        return None
    return {"items": sum(1 for r in results if r["present_n"]), "tests": pn + on,
            "pass_rate": round((p + o) / (pn + on), 4), "present_rate": round(p / pn, 4),
            "order_rate": round(o / on, 4) if on else None}
