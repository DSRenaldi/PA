import pandas as pd

from rule_helpers import (
    and_operator,
    clean_tokens,
    get_sentiment_from_preposition,
    get_sentiment_from_verb,
    is_adjective,
    is_probable_verb,
    is_sentiment_preposition,
    to_label,
)


def classify_rule_4_preposisi_verba(tokens):
    clean = clean_tokens(tokens)

    candidate_pairs = []
    for i in range(len(clean) - 1):
        left_token = clean[i]
        right_token = clean[i + 1]

        if is_sentiment_preposition(left_token) and is_probable_verb(right_token):
            left_score = get_sentiment_from_preposition(left_token)
            right_score = get_sentiment_from_verb(right_token)
            if left_score != 0 and right_score != 0:
                candidate_pairs.append((left_token, right_token, left_score, right_score))

    adjective_hits = [token for token in clean if is_adjective(token)]

    if len(candidate_pairs) == 1 and not adjective_hits:
        preposition_token, verb_token, preposition_score, verb_score = candidate_pairs[0]
        sentiment_value = and_operator(preposition_score, verb_score)
        return pd.Series(
            {
                "rule_4_applied": True,
                "rule_4_preposition": preposition_token,
                "rule_4_verb": verb_token,
                "rule_4_preposition_score": preposition_score,
                "rule_4_verb_score": verb_score,
                "rule_4_adjectives": adjective_hits,
                "rule_4_sentiment_value": sentiment_value,
                "rule_4_sentiment_label": to_label(sentiment_value),
            }
        )

    return pd.Series(
        {
            "rule_4_applied": False,
            "rule_4_preposition": None,
            "rule_4_verb": None,
            "rule_4_preposition_score": 0,
            "rule_4_verb_score": 0,
            "rule_4_adjectives": adjective_hits,
            "rule_4_sentiment_value": 0,
            "rule_4_sentiment_label": "Tidak terdeteksi",
        }
    )


def apply_rule_4(df, token_column="tokens_normalized"):
    result = df[token_column].apply(classify_rule_4_preposisi_verba)
    return pd.concat([df, result], axis=1)
