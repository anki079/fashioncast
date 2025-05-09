# src/fashioncast/season_code.py
import re
from typing import NamedTuple, Optional


class ParsedSeason(NamedTuple):
    season_code: str  # "SS2018" / "FW2020"
    collection_type: str  # "ready to wear" / "menswear" / "couture" / None
    raw_text: str


YEAR_RGX = re.compile(r"\b(19|20)\d{2}\b")
SPRING_RGX = re.compile(r"\b(spring|resort|cruise)\b", re.I)
FALL_RGX = re.compile(r"\b(fall|autumn|pre[- ]?fall)\b", re.I)
COUTURE_RGX = re.compile(r"\bcouture\b", re.I)
MENS_RGX = re.compile(r"\bmenswear\b", re.I)
RTW_RGX = re.compile(r"ready to wear", re.I)
JAN_RGX = re.compile(r"\b(jan|january)\b", re.I)
JUL_RGX = re.compile(r"\b(jul|july)\b", re.I)


def canonical_season(text: str) -> Optional[ParsedSeason]:
    """Map 'chanel,fall 2011 couture' → ParsedSeason('FW2011','couture',raw)"""
    t = text.lower()

    # year -----------------------------------------------------------
    m = YEAR_RGX.search(t)
    if not m:
        return None
    year = m.group(0)

    # collection type -----------------------------------------------
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
    elif ctype == "couture":
        season_code = f"{'SS' if JAN_RGX.search(t) else 'FW'}{year}"
    else:  # fallback (rare)
        season_code = f"SS{year}"

    return ParsedSeason(season_code, ctype, text)
