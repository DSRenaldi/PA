import ast
from pathlib import Path

import pandas as pd

from rule_helpers import BASE_DIR, PREPOSITION_SENTIMENT


def _load_slang_dictionary():
    path = BASE_DIR / "update_combined_slang_words.txt"
    return ast.literal_eval(path.read_text(encoding="utf-8"))


def _load_stopwords():
    path = BASE_DIR / "combined_stop_words.txt"
    return {line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _load_indonesian_dictionary():
    dictionary_words = set()

    slang_df = pd.read_csv(Path(r"d:\Kuliah\PA\Najma\colloquial-indonesian-lexicon.csv"), sep=";")
    slang_df["formal"] = slang_df["formal"].astype(str).str.lower().str.strip()
    slang_df["in_dict_flag"] = slang_df["In-dictionary"].astype(str).str.strip()
    dictionary_words.update(slang_df.loc[slang_df["in_dict_flag"] == "1", "formal"].tolist())

    slang_map = _load_slang_dictionary()
    dictionary_words.update(str(key).lower().strip() for key in slang_map.keys())
    dictionary_words.update(str(value).lower().strip() for value in slang_map.values())

    stop_words = _load_stopwords()
    dictionary_words.update(stop_words)

    for lexicon_name in ["positive.tsv", "negative.tsv"]:
        lexicon_df = pd.read_csv(BASE_DIR / lexicon_name, sep="\t")
        words = lexicon_df["word"].astype(str).str.lower().str.strip().tolist()
        for word in words:
            dictionary_words.update(part for part in word.split() if part)

    dictionary_words.update(PREPOSITION_SENTIMENT.keys())
    dictionary_words.update(
        {
            "terima",
            "kasih",
            "air",
            "mata",
            "minta",
            "tolong",
        }
    )

    return dictionary_words


INDONESIAN_DICTIONARY = _load_indonesian_dictionary()
KEEP_TOKENS = {".", ",", "?", "!"}


def split_valid_and_ignored_tokens(tokens):
    if not isinstance(tokens, list):
        return [], []

    valid_tokens = []
    ignored_tokens = []

    for token in tokens:
        normalized = str(token).lower().strip()
        if not normalized:
            continue

        if normalized in KEEP_TOKENS:
            valid_tokens.append(normalized)
        elif normalized in INDONESIAN_DICTIONARY:
            valid_tokens.append(normalized)
        else:
            ignored_tokens.append(normalized)

    return valid_tokens, ignored_tokens


def apply_rule_11(df, token_column="tokens_normalized"):
    result = df[token_column].apply(split_valid_and_ignored_tokens)

    df = df.copy()
    df["rule_11_valid_tokens"] = result.apply(lambda item: item[0])
    df["rule_11_ignored_tokens"] = result.apply(lambda item: item[1])
    df["rule_11_text"] = df["rule_11_valid_tokens"].apply(lambda tokens: " ".join(tokens))
    df["rule_11_ignored_count"] = df["rule_11_ignored_tokens"].apply(len)
    return df
