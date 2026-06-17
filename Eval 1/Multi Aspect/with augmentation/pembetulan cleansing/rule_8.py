import re

import pandas as pd

from rule_helpers import to_label


def split_sentences_by_dot_comma(text):
    if pd.isna(text) or not isinstance(text, str):
        return []

    parts = re.split(r"[.,]+", text)
    sentences = [part.strip() for part in parts if part.strip()]
    return sentences


def sum_sentence_scores(scores):
    total_score = sum(score for score in scores if isinstance(score, (int, float)))
    return total_score, to_label(total_score)


def apply_rule_8(df, text_column="normalized_text"):
    result = df[text_column].apply(split_sentences_by_dot_comma)

    df = df.copy()
    df["rule_8_sentences"] = result
    df["rule_8_sentence_count"] = df["rule_8_sentences"].apply(len)
    df["rule_8_sentence_1"] = df["rule_8_sentences"].apply(lambda items: items[0] if len(items) > 0 else "")
    df["rule_8_sentence_2"] = df["rule_8_sentences"].apply(lambda items: items[1] if len(items) > 1 else "")
    return df
