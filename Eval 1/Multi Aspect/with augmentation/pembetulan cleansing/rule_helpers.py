import re
from pathlib import Path

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


factory = StemmerFactory()
stemmer = factory.create_stemmer()

BASE_DIR = Path(__file__).resolve().parent

VERB_SENTIMENT_OVERRIDE = {
    "merugikan": -1,
    "rugi": -1,
    "menguntungkan": 1,
    "untung": 1,
    "menangani": 1,
    "tangani": 1,
    "menindaklanjuti": 1,
    "ditindaklanjuti": 1,
    "mengalir": 1,
    "alir": 1,
}

ADJECTIVE_SENTIMENT_OVERRIDE = {
    "kecewa": -1,
    "lama": -1,
    "cepat": 1,
    "lambat": -1,
    "bagus": 1,
    "baik": 1,
    "buruk": -1,
    "jelek": -1,
    "ramah": 1,
    "parah": -1,
    "keruh": -1,
    "bersih": 1,
    "kotor": -1,
    "jernih": 1,
    "mahal": -1,
    "murah": 1,
    "normal": 1,
    "puas": 1,
}

PREPOSITION_SENTIMENT = {
    "sangat": 1,
    "amat": 1,
    "sekali": 1,
    "terlalu": 1,
    "cukup": 1,
    "lebih": 1,
    "tidak": -1,
    "tak": -1,
    "bukan": -1,
    "belum": -1,
    "kurang": -1,
    "ga": -1,
    "gak": -1,
    "nggak": -1,
    "enggak": -1,
    "jangan": -1,
}

ADJECTIVE_WORDS = {
    "baik", "buruk", "bagus", "jelek", "parah", "cepat", "lambat", "ramah",
    "bersih", "kotor", "keruh", "jernih", "mahal", "murah", "sulit", "mudah",
    "normal", "kecewa", "puas", "lemah", "kuat", "lama", "baru",
}
ADJECTIVE_STEMS = {stemmer.stem(word) for word in ADJECTIVE_WORDS}

VERB_SEED_WORDS = {
    "alir", "respon", "tanggap", "layan", "bantu", "rugi", "untung", "bayar",
    "keluh", "lapor", "adu", "cek", "perbaiki", "tindak", "nyala", "naik",
    "turun", "putus", "bocor", "macet", "ganti", "kirim", "sambung", "tangani",
}


def _build_sentiment_lexicon():
    positive_lexicon = pd.read_csv(BASE_DIR / "positive.tsv", sep="\t", encoding="utf-8")
    negative_lexicon = pd.read_csv(BASE_DIR / "negative.tsv", sep="\t", encoding="utf-8")

    lexicon_df = pd.concat([positive_lexicon, negative_lexicon], ignore_index=True)
    lexicon_df["word"] = lexicon_df["word"].astype(str).str.lower().str.strip()
    lexicon_df["stemmed_word"] = lexicon_df["word"].apply(stemmer.stem)

    return (
        lexicon_df.groupby("stemmed_word", as_index=False)["weight"]
        .sum()
        .set_index("stemmed_word")["weight"]
        .to_dict()
    )


SENTIMENT_LEXICON = _build_sentiment_lexicon()


def normalize_alpha_token(token):
    token = str(token).lower().strip()
    return re.sub(r"[^a-z]", "", token)


def to_label(score):
    if score > 0:
        return "Positif"
    if score < 0:
        return "Negatif"
    return "Netral"


def and_operator(left_score, right_score):
    if left_score == 0 or right_score == 0:
        return 0
    return left_score * right_score


def nand_operator(left_score, right_score):
    if left_score == 0 or right_score == 0:
        return 0
    if left_score == 1 and right_score == 1:
        return 1
    return -1


def _lookup_sentiment(token, override_map):
    token = normalize_alpha_token(token)
    stem = stemmer.stem(token)

    if token in override_map:
        return override_map[token]
    if stem in override_map:
        return override_map[stem]

    raw_score = SENTIMENT_LEXICON.get(stem, 0)
    if raw_score > 0:
        return 1
    if raw_score < 0:
        return -1
    return 0


def get_sentiment_from_adjective(token):
    return _lookup_sentiment(token, ADJECTIVE_SENTIMENT_OVERRIDE)


def get_sentiment_from_verb(token):
    return _lookup_sentiment(token, VERB_SENTIMENT_OVERRIDE)


def get_sentiment_from_preposition(token):
    token = normalize_alpha_token(token)
    return PREPOSITION_SENTIMENT.get(token, 0)


def is_sentiment_preposition(token):
    token = normalize_alpha_token(token)
    return token in PREPOSITION_SENTIMENT


def is_adjective(token):
    token = normalize_alpha_token(token)
    if not token:
        return False
    return stemmer.stem(token) in ADJECTIVE_STEMS


def is_probable_verb(token):
    token = normalize_alpha_token(token)
    if not token:
        return False

    stem = stemmer.stem(token)
    if is_sentiment_preposition(token) or stem in ADJECTIVE_STEMS:
        return False

    verb_prefixes = ("me", "mem", "men", "meng", "meny", "ber", "ter", "di", "pe")
    verb_suffixes = ("kan", "i")

    return (
        token.startswith(verb_prefixes)
        or token.endswith(verb_suffixes)
        or stem in VERB_SEED_WORDS
    )


def clean_tokens(tokens):
    if not isinstance(tokens, list):
        return []
    cleaned = [normalize_alpha_token(token) for token in tokens]
    return [token for token in cleaned if token]
