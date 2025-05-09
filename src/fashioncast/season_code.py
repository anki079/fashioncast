import re
from typing import NamedTuple, Optional


# ------------------------------------------------------------------ #
#  Data class returned by the parser
# ------------------------------------------------------------------ #
class ParsedSeason(NamedTuple):
    season_code: str  # "SS2018" / "FW2020"
    collection_type: str  # "ready to wear" / "menswear" / "couture" / None
    raw_text: str


# ------------------------------------------------------------------ #
#  Regex caches
# ------------------------------------------------------------------ #
# short codes like SS2019 / FW2020 / PF2021 / RS2018 / HC2023 / MS2022
CODE_RGX = re.compile(r"^(SS|FW|PF|RS|CR|HC|MS)(\d{4})$", re.I)

YEAR_RGX = re.compile(r"\b(19|20)\d{2}\b")
SPRING_RGX = re.compile(r"\b(spring|resort|cruise)\b", re.I)
FALL_RGX = re.compile(r"\b(fall|autumn|pre[- ]?fall)\b", re.I)
COUTURE_RGX = re.compile(r"\bcouture\b", re.I)
MENS_RGX = re.compile(r"\bmenswear\b", re.I)
RTW_RGX = re.compile(r"ready to wear", re.I)
JAN_RGX = re.compile(r"\b(jan|january)\b", re.I)
JUL_RGX = re.compile(r"\b(jul|july)\b", re.I)


# ------------------------------------------------------------------ #
#  Helper for the two-letter Vogue codes
# ------------------------------------------------------------------ #
def _from_code(code: str) -> Optional[ParsedSeason]:
    m = CODE_RGX.match(code.strip())
    if not m:
        return None
    prefix, year = m.group(1).upper(), m.group(2)

    # Map extra prefixes onto SS/FW buckets
    if prefix in ("SS", "RS", "CR"):  # Resort/Cruise → Spring/Summer
        season_code = f"SS{year}"
    elif prefix in ("FW", "PF"):  # Pre-Fall → Fall/Winter
        season_code = f"FW{year}"
    elif prefix == "HC":  # Haute Couture: Jan=>SS, Jul=>FW (unknown → SS)
        season_code = f"SS{year}"
    elif prefix == "MS":  # Menswear; assume same season letter
        season_code = f"{'SS' if 'S' in prefix else 'FW'}{year}"
    else:
        season_code = f"SS{year}"

    return ParsedSeason(season_code, None, code)


# ------------------------------------------------------------------ #
#  Main public function
# ------------------------------------------------------------------ #
def canonical_season(text: str) -> Optional[ParsedSeason]:
    """
    Accept either:
      • short code  (e.g. 'RS2019', 'PF2020', 'FW2018')
      • free text   ('chanel, fall 2011 couture')
    Return ParsedSeason(season_code, collection_type, raw_text)
    """
    # 1) fast-path: already a code
    parsed = _from_code(text)
    if parsed:
        return parsed

    # 2) free-text path
    t = text.lower()

    # year -----------------------------------------------------------
    m = YEAR_RGX.search(t)
    if not m:
        return None
    year = m.group(0)

    # collection type ------------------------------------------------
    if COUTURE_RGX.search(t):
        ctype = "couture"
    elif MENS_RGX.search(t):
        ctype = "menswear"
    elif RTW_RGX.search(t):
        ctype = "ready to wear"
    else:
        ctype = None

    # season → SS / FW ----------------------------------------------
    if SPRING_RGX.search(t):
        season_code = f"SS{year}"
    elif FALL_RGX.search(t):
        season_code = f"FW{year}"
    elif ctype == "couture":  # couture without explicit 'spring/fall'
        season_code = f"{'SS' if JAN_RGX.search(t) else 'FW'}{year}"
    else:  # conservative fallback
        season_code = f"SS{year}"

    return ParsedSeason(season_code, ctype, text)
