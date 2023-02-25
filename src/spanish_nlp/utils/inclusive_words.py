"""
Inclusive words data dictonary
"""

INCLUSIVE_WORDS = {
    "nosotres": "nosotros",
    "nosotrxs": "nosotros",
    "nosotr@s": "nosotros",
    "todes": "todos",
    "todxs": "todos",
    "tod@s": "todos",
    "amiges": "amigos",
    "amigxs": "amigos",
    "amig@s": "amigos",
    "compañeres": "compañeros",
    "compañerxs": "compañeros",
    "compañer@s": "compañeros",
    "chiques": "chicos",
    "chicxs": "chicos",
    "chic@s": "chicos",
    "niñes": "niños",
    "niñxs": "niños",
    "niñ@s": "niños",
    "elle": "él",
    "ellxs": "ellos",
    "ell@s": "ellos",
    "les": "los",
}

def normalize_inclusive_language(text: str) -> str:
    """
    Normalize inclusive language
    """
    for key, value in INCLUSIVE_WORDS.items():
        if key in text:
            text = text.replace(key, value)
    return text
