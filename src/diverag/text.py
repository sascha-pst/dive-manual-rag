"""Tokenizing, light stemming and the diver-slang synonym map."""
from __future__ import annotations

import re

STOP = set(
    "a an the of and or to in on at by for is are be as it its this that with from "
    "into than so if when do does did i my me you your what why how who which can could "
    "should would during before after there their them they not no any each about get "
    "getting go going while work out".split()
)

# Diver slang -> manual vocabulary. Used for query expansion by the lexical
# fallback "semantic" scorer. A real embedding model makes most of this
# unnecessary; keep it small and review it so it doesn't leak the eval set.
SYNONYMS: dict[str, str] = {
    "bends": "decompression sickness bubble",
    "dcs": "decompression sickness",
    "narked": "narcosis nitrogen",
    "drunk": "narcosis intoxication",
    "tipsy": "narcosis intoxication",
    "embolism": "arterial gas embolism",
    "squeeze": "barotrauma equalize",
    "ear": "barotrauma equalize middle",
    "ears": "barotrauma equalize middle",
    "hurt": "pain",
    "fly": "flying altitude aircraft",
    "flying": "altitude aircraft",
    "plane": "flying aircraft altitude",
    "airplane": "flying aircraft altitude",
    "seizure": "convulsion oxygen toxicity",
    "convulsion": "oxygen toxicity",
    "o2": "oxygen",
    "po2": "partial pressure oxygen",
    "deep": "depth pressure",
    "chamber": "recompression treatment",
    "sewage": "contaminated polluted",
    "polluted": "contaminated",
    "dirty": "contaminated polluted",
    "gear": "equipment",
    "check": "checks before",
    "regulator": "stage cylinder",
    "up": "ascent ascend",
    "surfacing": "ascent",
    "descending": "descent equalize",
    "volume": "boyle expand",
    "picks": "selection directed",
    "wait": "interval",
    "water": "entering",
    "breath": "holding lungs",
}


def stem(w: str) -> str:
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def expand(query: str) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", query.lower())
    extra = " ".join(SYNONYMS[w] for w in raw if w in SYNONYMS)
    return tokenize(f"{query} {extra}")
