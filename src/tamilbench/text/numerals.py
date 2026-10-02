"""Tamil numerals, calendar/accounting signs and the Tamil Supplement fractions.

Tamil numerals survive in palm-leaf horoscopes, land and revenue records, temple
accounts, old print (page numbers, dates) and inscriptions. Two systems are in use:

  positional   ௧௯௪௭              (1947; the zero ௦ is a later addition)
  traditional  ௲௯௱௪௰௭           (1000 + 9×100 + 4×10 + 7; ௰ ௱ ௲ are 10, 100, 1000)
"""

from __future__ import annotations

import random

TAMIL_DIGITS = "௦௧௨௩௪௫௬௭௮௯"
TEN, HUNDRED, THOUSAND = "௰", "௱", "௲"
DAY, MONTH, YEAR, DEBIT, CREDIT, AS_ABOVE, RUPEE, NUMBER = "௳௴௵௶௷௸௹௺"

# Tamil Supplement (U+11FC0–U+11FFF, Unicode 12): fractions, measures and account signs.
FRACTIONS = {
    "1/320": "\U00011FC0", "1/160": "\U00011FC1", "1/80": "\U00011FC2", "1/64": "\U00011FC3",
    "1/40": "\U00011FC4", "1/32": "\U00011FC5", "3/80": "\U00011FC6", "3/64": "\U00011FC7",
    "1/20": "\U00011FC8", "1/16": "\U00011FC9", "1/10": "\U00011FCB", "1/8": "\U00011FCC",
    "3/20": "\U00011FCD", "3/16": "\U00011FCE", "1/5": "\U00011FCF", "1/4": "\U00011FD0",
    "1/2": "\U00011FD1", "3/4": "\U00011FD3",
}
MEASURES = {  # grain/volume, money, land
    "nel": "\U00011FD5", "cevitu": "\U00011FD6", "aazhaakku": "\U00011FD7", "uzhakku": "\U00011FD8",
    "muuvuzhakku": "\U00011FD9", "kuruni": "\U00011FDA", "pathakku": "\U00011FDB",
    "mukkuruni": "\U00011FDC", "kaacu": "\U00011FDD", "panam": "\U00011FDE", "pon": "\U00011FDF",
    "varaakan": "\U00011FE0", "paaram": "\U00011FE1", "kuzhi": "\U00011FE2", "veli": "\U00011FE3",
}
ACCOUNT_SIGNS = {
    "wet-cultivation": "\U00011FE4", "dry-cultivation": "\U00011FE5", "land": "\U00011FE6",
    "salt-pan": "\U00011FE7", "credit": "\U00011FE8", "number": "\U00011FE9",
    "current": "\U00011FEA", "and-odd": "\U00011FEB", "spent": "\U00011FEC",
    "total": "\U00011FED", "in-possession": "\U00011FEE", "starting-from": "\U00011FEF",
    "muthaliya": "\U00011FF0", "vakaiyaraa": "\U00011FF1",
}
END_OF_TEXT = "\U00011FFF"

TAMIL_MONTHS = ["சித்திரை", "வைகாசி", "ஆனி", "ஆடி", "ஆவணி", "புரட்டாசி",
                "ஐப்பசி", "கார்த்திகை", "மார்கழி", "தை", "மாசி", "பங்குனி"]


def to_positional(n: int) -> str:
    """>>> to_positional(1947)
    '௧௯௪௭'"""
    if n < 0:
        raise ValueError("negative")
    return "".join(TAMIL_DIGITS[int(d)] for d in str(n))


def to_traditional(n: int) -> str:
    """Additive Tamil numeral with ௰ ௱ ௲ (1 ≤ n < 100 000).

    >>> to_traditional(1947)
    '௲௯௱௪௰௭'
    >>> to_traditional(10)
    '௰'
    >>> to_traditional(20005)
    '௨௰௲௫'
    """
    if not 0 < n < 100_000:
        raise ValueError("traditional numerals supported for 1..99999")
    out = []
    th, rest = divmod(n, 1000)
    if th:
        out.append(("" if th == 1 else _below_thousand(th)) + THOUSAND)
    if rest:
        out.append(_below_thousand(rest))
    return "".join(out)


def _below_thousand(n: int) -> str:
    out = []
    h, rest = divmod(n, 100)
    if h:
        out.append(("" if h == 1 else TAMIL_DIGITS[h]) + HUNDRED)
    t, u = divmod(rest, 10)
    if t:
        out.append(("" if t == 1 else TAMIL_DIGITS[t]) + TEN)
    if u:
        out.append(TAMIL_DIGITS[u])
    return "".join(out)


def random_numeral_line(rng: random.Random) -> tuple[str, str]:
    """A line in the style of an old ledger, deed or horoscope. Returns (text, kind)."""
    kind = rng.choice(["date", "date", "amount", "ledger", "fraction", "measure", "count", "page"])
    num = to_traditional if rng.random() < 0.5 else to_positional
    if kind == "date":
        y = rng.randint(1700, 1977)
        text = f"{num(y)} {YEAR} {rng.choice(TAMIL_MONTHS)} {MONTH} {num(rng.randint(1, 31))} {DAY}"
    elif kind == "amount":
        amt = rng.randint(1, 9999)
        text = f"{RUPEE} {num(amt)}" + (f" {rng.choice(list(FRACTIONS.values()))}" if rng.random() < 0.4 else "")
    elif kind == "ledger":
        items = ["அரிசி", "நெல்", "எண்ணெய்", "துணி", "வரி", "கூலி", "குத்தகை", "விதை", "உப்பு", "பருப்பு"]
        sign = rng.choice([DEBIT, CREDIT, AS_ABOVE])
        text = f"{rng.choice(items)} {sign} {RUPEE} {num(rng.randint(1, 999))}"
    elif kind == "fraction":
        parts = rng.sample(list(FRACTIONS.values()), k=rng.randint(1, 3))
        text = f"{num(rng.randint(1, 999))} " + " ".join(parts)
    elif kind == "measure":
        unit = rng.choice(list(MEASURES.values()))
        text = f"{num(rng.randint(1, 500))} {unit}"
        if rng.random() < 0.5:
            text += f" {rng.choice(list(FRACTIONS.values()))}"
        if rng.random() < 0.4:
            text = f"{rng.choice(list(ACCOUNT_SIGNS.values()))} " + text
    elif kind == "count":
        text = f"{NUMBER} {num(rng.randint(1, 9999))}"
    else:  # page / leaf number
        text = f"ஏடு {num(rng.randint(1, 400))}" if rng.random() < 0.5 else f"பக்கம் {num(rng.randint(1, 999))}"
    return text, kind
