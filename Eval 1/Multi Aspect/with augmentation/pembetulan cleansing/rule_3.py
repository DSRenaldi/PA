import pandas as pd

from rule_helpers import (
    and_operator,
    clean_tokens,
    get_sentiment_from_adjective,
    get_sentiment_from_preposition,
    is_adjective,
    is_probable_verb,
    is_sentiment_preposition,
    to_label,
)


def classify_rule_3_preposisi_adjektif(tokens):
    clean = clean_tokens(tokens)

    candidate_pairs = []
    for i in range(len(clean) - 1):
        left_token = clean[i]
        right_token = clean[i + 1]

        if is_sentiment_preposition(left_token) and is_adjective(right_token):
            left_score = get_sentiment_from_preposition(left_token)
            right_score = get_sentiment_from_adjective(right_token)
            if left_score != 0 and right_score != 0:
                candidate_pairs.append((left_token, right_token, left_score, right_score))

    verb_hits = [token for token in clean if is_probable_verb(token)]

    if len(candidate_pairs) == 1 and not verb_hits:
        preposition_token, adjective_token, preposition_score, adjective_score = candidate_pairs[0]
        sentiment_value = and_operator(preposition_score, adjective_score)
        return pd.Series(
            {
                "rule_3_applied": True,
                "rule_3_preposition": preposition_token,
                "rule_3_adjective": adjective_token,
                "rule_3_preposition_score": preposition_score,
                "rule_3_adjective_score": adjective_score,
                "rule_3_verbs": verb_hits,
                "rule_3_sentiment_value": sentiment_value,
                "rule_3_sentiment_label": to_label(sentiment_value),
            }
        )

    return pd.Series(
        {
            "rule_3_applied": False,
            "rule_3_preposition": None,
            "rule_3_adjective": None,
            "rule_3_preposition_score": 0,
            "rule_3_adjective_score": 0,
            "rule_3_verbs": verb_hits,
            "rule_3_sentiment_value": 0,
            "rule_3_sentiment_label": "Tidak terdeteksi",
        }
    )


def apply_rule_3(df, token_column="tokens_normalized"):
    result = df[token_column].apply(classify_rule_3_preposisi_adjektif)
    return pd.concat([df, result], axis=1)
