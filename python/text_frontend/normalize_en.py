"""English text normalization without inflect/g2p_en/nltk.

The rule set follows the GPT-SoVITS English normalization intent (numbers,
dates, times, currency, units, ordinals, fractions, accents), but the
number-to-words and ordinal tables are implemented locally so the frontend has
no mandatory third-party English dependency.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

_UNITS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = ["", "thousand", "million", "billion", "trillion", "quadrillion"]

_ORDINAL_IRREGULAR = {
    "zero": "zeroth", "one": "first", "two": "second", "three": "third",
    "four": "fourth", "five": "fifth", "six": "sixth", "seven": "seventh",
    "eight": "eighth", "nine": "ninth", "ten": "tenth", "eleven": "eleventh",
    "twelve": "twelfth", "thirteen": "thirteenth", "fourteen": "fourteenth",
    "fifteen": "fifteenth", "sixteen": "sixteenth", "seventeen": "seventeenth",
    "eighteen": "eighteenth", "nineteen": "nineteenth", "twenty": "twentieth",
    "thirty": "thirtieth", "forty": "fortieth", "fifty": "fiftieth",
    "sixty": "sixtieth", "seventy": "seventieth", "eighty": "eightieth",
    "ninety": "ninetieth", "hundred": "hundredth", "thousand": "thousandth",
    "million": "millionth", "billion": "billionth", "trillion": "trillionth",
    "quadrillion": "quadrillionth",
}

# Measurement units: symbol -> (singular, plural).  Longest symbols first.
_MEASUREMENT = {
    "km/h": ("kilometer per hour", "kilometers per hour"),
    "°C": ("degree celsius", "degrees celsius"),
    "°F": ("degree fahrenheit", "degrees fahrenheit"),
    "tbsp": ("tablespoon", "tablespoons"),
    "tsp": ("teaspoon", "teaspoons"),
    "min": ("minute", "minutes"),
    "km": ("kilometer", "kilometers"),
    "kg": ("kilogram", "kilograms"),
    "ml": ("milliliter", "milliliters"),
    "cm": ("centimeter", "centimeters"),
    "mm": ("millimeter", "millimeters"),
    "ft": ("foot", "feet"),
    "in": ("inch", "inches"),
    "yd": ("yard", "yards"),
    "mi": ("mile", "miles"),
    "lb": ("pound", "pounds"),
    "oz": ("ounce", "ounces"),
    "ms": ("millisecond", "milliseconds"),
    "h": ("hour", "hours"),
    "m": ("meter", "meters"),
    "s": ("second", "seconds"),
    "L": ("liter", "liters"),
}
_MEASUREMENT_RE = re.compile(
    r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*("
    + "|".join(re.escape(unit) for unit in sorted(_MEASUREMENT, key=len, reverse=True))
    + r")(?![\w])"
)

_TIME_RE = re.compile(r"\b([01]?[0-9]|2[0-3]):([0-5][0-9])\b")
_ORDINAL_WORD_RE = re.compile(r"\b([0-9]+)(st|nd|rd|th)\b", re.IGNORECASE)
_ORDINAL_DOT_RE = re.compile(r"\b([0-9]+)\.\s+")
_FRACTION_RE = re.compile(r"(?<![\d.])(-?)([0-9]+)/([0-9]+)(?![\d.])")
_PERCENT_RE = re.compile(r"(-?)([0-9]+(?:\.[0-9]+)?)\s*%")
_DOLLARS_RE = re.compile(r"\$([0-9][0-9,]*(?:\.[0-9]+)?)")
_POUNDS_RE = re.compile(r"£([0-9][0-9,]*(?:\.[0-9]+)?)")
_DECIMAL_RE = re.compile(r"(?<![\d.])(-?)([0-9]+)\.([0-9]+)(?![\d.])")
_COMMA_NUMBER_RE = re.compile(r"([0-9][0-9,]*[0-9])")
_NUMBER_RE = re.compile(r"(?<![\w.])[0-9]+(?![\w.])")

_FULLWIDTH = {ord(ch): ord(ch) - 0xFEE0 for ch in map(chr, range(0xFF01, 0xFF5F))}
_CJK_PUNCT = {
    "，": ",", "。": ".", "！": "!", "？": "?", "；": ",", "：": ",",
    "、": ",", "—": "-", "–": "-", "～": "~", "…": "...", "“": '"',
    "”": '"', "‘": "'", "’": "'", "【": " ", "】": " ", "《": " ", "》": " ",
    "（": "(", "）": ")", "￥": "¥",
}


def _three_digits_to_words(value: int) -> str:
    words: List[str] = []
    hundreds, rest = divmod(value, 100)
    if hundreds:
        words.extend([_UNITS[hundreds], "hundred"])
    if rest:
        if rest < 20:
            words.append(_UNITS[rest])
        else:
            tens, unit = divmod(rest, 10)
            words.append(_TENS[tens] + (f"-{_UNITS[unit]}" if unit else ""))
    return " ".join(words)


def number_to_words(value: int) -> str:
    """Cardinal number words for non-negative integers up to quadrillions."""
    if value < 0:
        return "minus " + number_to_words(-value)
    if value == 0:
        return "zero"
    groups: List[str] = []
    scale_index = 0
    while value:
        value, chunk = divmod(value, 1000)
        if chunk:
            chunk_words = _three_digits_to_words(chunk)
            if scale_index:
                chunk_words += " " + _SCALES[scale_index]
            groups.append(chunk_words)
        scale_index += 1
    return " ".join(reversed(groups))


def _ordinal_word(word: str) -> str:
    if word in _ORDINAL_IRREGULAR:
        return _ORDINAL_IRREGULAR[word]
    if word.endswith("y"):
        return word[:-1] + "ieth"
    return word + "th"


def number_to_ordinal_words(value: int) -> str:
    if value < 0:
        return "minus " + number_to_ordinal_words(-value)
    cardinal = number_to_words(value)
    if "-" in cardinal:
        head, tail = cardinal.rsplit("-", 1)
        return head + "-" + _ordinal_word(tail)
    parts = cardinal.split()
    parts[-1] = _ordinal_word(parts[-1])
    return " ".join(parts)


def _year_like(value: int) -> str:
    """Read 1001..2999 like a year, matching the GPT-SoVITS behavior."""
    if not (1000 < value < 3000):
        return number_to_words(value)
    if value == 2000:
        return "two thousand"
    if 2000 < value < 2010:
        return "two thousand " + number_to_words(value % 100)
    high, low = divmod(value, 100)
    high_words = number_to_words(high)
    if low == 0:
        return high_words + " hundred"
    if low < 10:
        return high_words + " oh " + number_to_words(low)
    return high_words + " " + number_to_words(low)


def _expand_time(match: re.Match) -> str:
    hours, minutes = int(match.group(1)), int(match.group(2))
    period = "a.m." if hours < 12 else "p.m."
    if hours == 0:
        hours = 12
    elif hours > 12:
        hours -= 12
    hour_words = number_to_words(hours)
    if minutes == 0:
        return f"{hour_words} o'clock {period}"
    return f"{hour_words} {number_to_words(minutes)} {period}"


def _expand_measurement(match: re.Match) -> str:
    number_text, unit = match.group(1), match.group(2)
    singular, plural = _MEASUREMENT[unit]
    try:
        is_one = float(number_text) == 1.0
    except ValueError:
        is_one = False
    return f"{_expand_numeric(number_text)} {singular if is_one else plural}"


def _expand_currency(number_text: str, major_singular: str, major_plural: str,
                     minor_singular: str, minor_plural: str) -> str:
    number_text = number_text.replace(",", "")
    if "." in number_text:
        major_text, minor_text = number_text.split(".", 1)
    else:
        major_text, minor_text = number_text, ""
    major = int(major_text) if major_text else 0
    minor = int(minor_text.ljust(2, "0")[:2]) if minor_text else 0
    if major and minor:
        return (
            f"{number_to_words(major)} {major_singular if major == 1 else major_plural} "
            f"and {number_to_words(minor)} {minor_singular if minor == 1 else minor_plural}"
        )
    if major:
        return f"{number_to_words(major)} {major_singular if major == 1 else major_plural}"
    if minor:
        return f"{number_to_words(minor)} {minor_singular if minor == 1 else minor_plural}"
    return f"zero {major_plural}"


def _expand_decimal(match: re.Match) -> str:
    sign, integer, decimal = match.group(1), match.group(2), match.group(3)
    prefix = "minus " if sign else ""
    digits = " ".join(_UNITS[int(ch)] for ch in decimal)
    return f"{prefix}{number_to_words(int(integer))} point {digits}"


def _expand_fraction(match: re.Match) -> str:
    sign, numerator_text, denominator_text = match.group(1), match.group(2), match.group(3)
    numerator, denominator = int(numerator_text), int(denominator_text)
    prefix = "minus " if sign else ""
    if denominator == 0:
        return match.group(0)
    numerator_words = number_to_words(numerator)
    if denominator == 2:
        denominator_words = "half" if numerator == 1 else "halves"
    elif denominator == 1:
        return prefix + numerator_words
    else:
        denominator_words = number_to_ordinal_words(denominator)
        if numerator != 1:
            denominator_words += "s"
    return f"{prefix}{numerator_words} {denominator_words}"


def _expand_percent(match: re.Match) -> str:
    sign, number_text = match.group(1), match.group(2)
    return ("minus " if sign else "") + _expand_numeric(number_text) + " percent"


def _expand_numeric(text: str) -> str:
    """Cardinal/decimal words for a plain numeric token."""
    text = text.replace(",", "")
    if "." in text:
        integer, decimal = text.split(".", 1)
        digits = " ".join(_UNITS[int(ch)] for ch in decimal if ch.isdigit())
        return f"{number_to_words(int(integer))} point {digits}"
    return _year_like(int(text))


def _remove_commas(match: re.Match) -> str:
    return match.group(1).replace(",", "")


def normalize_en(text: str, *, preserve_edge_spaces: bool = False) -> str:
    """Normalize English NSW and return a clean English/ARPA-friendly string.

    ``preserve_edge_spaces`` keeps a single leading/trailing space when the
    input had one.  Mixed-mode segments use this so concatenating a zh and en
    segment keeps the original word boundary (``你好 world`` rather than
    ``你好world``).
    """
    if not text:
        return ""
    leading = " " if preserve_edge_spaces and text[0].isspace() else ""
    trailing = " " if preserve_edge_spaces and text[-1].isspace() else ""

    # Full-width and CJK punctuation to ASCII.
    text = text.translate(_FULLWIDTH)
    for source, target in _CJK_PUNCT.items():
        text = text.replace(source, target)

    # Abbreviations that should not be read as punctuation.
    text = re.sub(r"(?i)\bi\.e\.", "that is", text)
    text = re.sub(r"(?i)\be\.g\.", "for example", text)

    text = re.sub(r"(?<!\d)\d{1,3}(?:,\d{3})+(?:\.\d+)?", lambda m: m.group().replace(",", ""), text)
    for symbol, spoken in {"€": " euros ", "¥": " yuan ", "&": " and ", "+": " plus ", "=": " equals ", "@": " at ", "#": " number ", "×": " times ", "÷": " divided by "}.items():
        text = text.replace(symbol, spoken)

    # Time before generic numbers.
    text = _TIME_RE.sub(_expand_time, text)

    # Currencies before generic numbers.
    text = _DOLLARS_RE.sub(
        lambda m: _expand_currency(m.group(1), "dollar", "dollars", "cent", "cents"), text
    )
    text = _POUNDS_RE.sub(
        lambda m: _expand_currency(m.group(1), "pound", "pounds", "penny", "pence"), text
    )
    text = re.sub(
        r"([0-9]+(?:\.[0-9]+)?)\$",
        lambda m: _expand_currency(m.group(1), "dollar", "dollars", "cent", "cents"), text,
    )
    text = re.sub(
        r"([0-9]+(?:\.[0-9]+)?)£",
        lambda m: _expand_currency(m.group(1), "pound", "pounds", "penny", "pence"), text,
    )

    # Units before generic numbers.
    text = _MEASUREMENT_RE.sub(_expand_measurement, text)

    # Fractions, percentages, decimals, ordinals.
    text = _FRACTION_RE.sub(_expand_fraction, text)
    text = _PERCENT_RE.sub(_expand_percent, text)
    text = _ORDINAL_WORD_RE.sub(lambda m: number_to_ordinal_words(int(m.group(1))), text)
    text = _DECIMAL_RE.sub(_expand_decimal, text)

    # Comma-grouped integers then bare integers.
    text = _COMMA_NUMBER_RE.sub(_remove_commas, text)
    text = re.sub(r"[0-9]+", lambda m: " " + _year_like(int(m.group(0))) + " ", text)
    text = re.sub(r"\s+([.,?!])", r"\1", text)

    # Strip accents (café -> cafe).
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )

    # Characters the English G2P can pronounce or pass through as punctuation.
    text = re.sub(r"[^ A-Za-z'.,?!\-]", " ", text)
    text = re.sub(r"(?<!^)(?<![\s])([A-Z])", r" \1", text)
    text = re.sub(r"([.,?!])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return leading + text + trailing
