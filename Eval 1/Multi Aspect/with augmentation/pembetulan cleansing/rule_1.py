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
}

PREPOSITIONS = {
    "di", "ke", "dari", "daripada", "pada", "untuk", "dengan", "dalam",
    "tanpa", "oleh", "kepada", "terhadap", "sejak", "hingga", "sampai",
    "antara", "demi", "tentang", "bagi",
}

ADJECTIVE_WORDS = {
    "baik", "buruk", "bagus", "jelek", "parah", "cepat", "lambat", "ramah",
    "bersih", "kotor", "keruh", "jernih", "mahal", "murah", "sulit", "mudah",
    "normal", "kecewa", "puas", "lemah", "kuat", "mati", "hidup",
}
ADJECTIVE_STEMS = {stemmer.stem(word) for word in ADJECTIVE_WORDS}

VERB_SEED_WORDS = {
    "alir", "respon", "tanggap", "layan", "bantu", "rugi", "untung", "bayar",
    "keluh", "lapor", "adu", "cek", "perbaiki", "tindak", "mati", "nyala",
    "naik", "turun", "putus", "bocor", "macet", "ganti", "kirim", "sambung",
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


def get_sentiment_from_verb(token):
    token = normalize_alpha_token(token)
    stem = stemmer.stem(token)

    if token in VERB_SENTIMENT_OVERRIDE:
        return VERB_SENTIMENT_OVERRIDE[token]
    if stem in VERB_SENTIMENT_OVERRIDE:
        return VERB_SENTIMENT_OVERRIDE[stem]

    raw_score = SENTIMENT_LEXICON.get(stem, 0)
    if raw_score > 0:
        return 1
    if raw_score < 0:
        return -1
    return 0


def is_adjective(token):
    token = normalize_alpha_token(token)
    if not token:
        return False
    return stemmer.stem(token) in ADJECTIVE_STEMS


def is_preposition(token):
    token = normalize_alpha_token(token)
    return token in PREPOSITIONS


def is_probable_verb(token):
    token = normalize_alpha_token(token)
    if not token:
        return False

    stem = stemmer.stem(token)
    if token in PREPOSITIONS or stem in ADJECTIVE_STEMS:
        return False

    verb_prefixes = ("me", "mem", "men", "meng", "meny", "ber", "ter", "di", "pe")
    verb_suffixes = ("kan", "i")

    return (
        token.startswith(verb_prefixes)
        or token.endswith(verb_suffixes)
        or stem in VERB_SEED_WORDS
    )


def classify_rule_1_verba_tunggal(tokens):
    if not isinstance(tokens, list):
        tokens = []

    clean_tokens = [normalize_alpha_token(token) for token in tokens]
    clean_tokens = [token for token in clean_tokens if token]

    preposition_hits = [token for token in clean_tokens if is_preposition(token)]
    adjective_hits = [token for token in clean_tokens if is_adjective(token)]

    verb_hits = []
    for token in clean_tokens:
        if is_probable_verb(token):
            score = get_sentiment_from_verb(token)
            if score != 0:
                verb_hits.append((token, score))

    if len(verb_hits) == 1 and not adjective_hits and not preposition_hits:
        verb_token, sentiment_value = verb_hits[0]
        sentiment_label = "Positif" if sentiment_value > 0 else "Negatif"
        return pd.Series(
            {
                "rule_1_applied": True,
                "rule_1_verb": verb_token,
                "rule_1_verb_score": sentiment_value,
                "rule_1_adjectives": adjective_hits,
                "rule_1_prepositions": preposition_hits,
                "rule_1_sentiment_value": sentiment_value,
                "rule_1_sentiment_label": sentiment_label,
            }
        )

    return pd.Series(
        {
            "rule_1_applied": False,
            "rule_1_verb": None,
            "rule_1_verb_score": 0,
            "rule_1_adjectives": adjective_hits,
            "rule_1_prepositions": preposition_hits,
            "rule_1_sentiment_value": 0,
            "rule_1_sentiment_label": "Tidak terdeteksi",
        }
    )


def apply_rule_1(df, token_column="tokens_normalized"):
    result = df[token_column].apply(classify_rule_1_verba_tunggal)
    return pd.concat([df, result], axis=1)

