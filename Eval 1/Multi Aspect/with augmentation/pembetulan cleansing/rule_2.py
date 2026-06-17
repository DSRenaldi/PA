import pandas as pd

from rule_helpers import (
    clean_tokens,
    get_sentiment_from_adjective,
    is_adjective,
    is_probable_verb,
    is_sentiment_preposition,
    to_label,
)


def classify_rule_2_adjektiva_tunggal(tokens):
    clean = clean_tokens(tokens)

    preposition_hits = [token for token in clean if is_sentiment_preposition(token)]
    verb_hits = [token for token in clean if is_probable_verb(token)]

    adjective_hits = []
    for token in clean:
        if is_adjective(token):
            score = get_sentiment_from_adjective(token)
            if score != 0:
                adjective_hits.append((token, score))

    if len(adjective_hits) == 1 and not verb_hits and not preposition_hits:
        adjective_token, sentiment_value = adjective_hits[0]
        return pd.Series(
            {
                "rule_2_applied": True,
                "rule_2_adjective": adjective_token,
                "rule_2_adjective_score": sentiment_value,
                "rule_2_verbs": verb_hits,
                "rule_2_prepositions": preposition_hits,
                "rule_2_sentiment_value": sentiment_value,
                "rule_2_sentiment_label": to_label(sentiment_value),
            }
        )

    return pd.Series(
        {
            "rule_2_applied": False,
            "rule_2_adjective": None,
            "rule_2_adjective_score": 0,
            "rule_2_verbs": verb_hits,
            "rule_2_prepositions": preposition_hits,
            "rule_2_sentiment_value": 0,
            "rule_2_sentiment_label": "Tidak terdeteksi",
        }
    )


def apply_rule_2(df, token_column="tokens_normalized"):
    result = df[token_column].apply(classify_rule_2_adjektiva_tunggal)
    return pd.concat([df, result], axis=1)
