import pandas as pd

from rule_helpers import (
    clean_tokens,
    get_sentiment_from_adjective,
    get_sentiment_from_verb,
    is_adjective,
    is_probable_verb,
    is_sentiment_preposition,
    nand_operator,
    to_label,
)


def classify_rule_5_verba_adjektif(tokens):
    clean = clean_tokens(tokens)

    candidate_pairs = []
    for i in range(len(clean) - 1):
        left_token = clean[i]
        right_token = clean[i + 1]

        if is_probable_verb(left_token) and is_adjective(right_token):
            verb_score = get_sentiment_from_verb(left_token)
            adjective_score = get_sentiment_from_adjective(right_token)
            if verb_score != 0 and adjective_score != 0:
                candidate_pairs.append((left_token, right_token, verb_score, adjective_score))

    preposition_hits = [token for token in clean if is_sentiment_preposition(token)]

    if len(candidate_pairs) == 1 and not preposition_hits:
        verb_token, adjective_token, verb_score, adjective_score = candidate_pairs[0]
        sentiment_value = nand_operator(verb_score, adjective_score)
        return pd.Series(
            {
                "rule_5_applied": True,
                "rule_5_verb": verb_token,
                "rule_5_adjective": adjective_token,
                "rule_5_verb_score": verb_score,
                "rule_5_adjective_score": adjective_score,
                "rule_5_prepositions": preposition_hits,
                "rule_5_sentiment_value": sentiment_value,
                "rule_5_sentiment_label": to_label(sentiment_value),
            }
        )

    return pd.Series(
        {
            "rule_5_applied": False,
            "rule_5_verb": None,
            "rule_5_adjective": None,
            "rule_5_verb_score": 0,
            "rule_5_adjective_score": 0,
            "rule_5_prepositions": preposition_hits,
            "rule_5_sentiment_value": 0,
            "rule_5_sentiment_label": "Tidak terdeteksi",
        }
    )


def apply_rule_5(df, token_column="tokens_normalized"):
    result = df[token_column].apply(classify_rule_5_verba_adjektif)
    return pd.concat([df, result], axis=1)
