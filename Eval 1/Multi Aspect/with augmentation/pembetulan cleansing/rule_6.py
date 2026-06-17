import pandas as pd

from rule_helpers import (
    and_operator,
    clean_tokens,
    get_sentiment_from_adjective,
    get_sentiment_from_preposition,
    get_sentiment_from_verb,
    is_adjective,
    is_probable_verb,
    is_sentiment_preposition,
    nand_operator,
    to_label,
)


def classify_rule_6_preposisi_verba_adjektif(tokens):
    clean = clean_tokens(tokens)

    candidate_triplets = []
    for i in range(len(clean) - 2):
        first_token = clean[i]
        second_token = clean[i + 1]
        third_token = clean[i + 2]

        if (
            is_sentiment_preposition(first_token)
            and is_probable_verb(second_token)
            and is_adjective(third_token)
        ):
            preposition_score = get_sentiment_from_preposition(first_token)
            verb_score = get_sentiment_from_verb(second_token)
            adjective_score = get_sentiment_from_adjective(third_token)

            if preposition_score != 0 and verb_score != 0 and adjective_score != 0:
                candidate_triplets.append(
                    (first_token, second_token, third_token, preposition_score, verb_score, adjective_score)
                )

    if len(candidate_triplets) == 1:
        (
            preposition_token,
            verb_token,
            adjective_token,
            preposition_score,
            verb_score,
            adjective_score,
        ) = candidate_triplets[0]

        verb_adjective_score = nand_operator(verb_score, adjective_score)
        sentiment_value = and_operator(preposition_score, verb_adjective_score)

        return pd.Series(
            {
                "rule_6_applied": True,
                "rule_6_preposition": preposition_token,
                "rule_6_verb": verb_token,
                "rule_6_adjective": adjective_token,
                "rule_6_preposition_score": preposition_score,
                "rule_6_verb_score": verb_score,
                "rule_6_adjective_score": adjective_score,
                "rule_6_verb_adjective_score": verb_adjective_score,
                "rule_6_sentiment_value": sentiment_value,
                "rule_6_sentiment_label": to_label(sentiment_value),
            }
        )

    return pd.Series(
        {
            "rule_6_applied": False,
            "rule_6_preposition": None,
            "rule_6_verb": None,
            "rule_6_adjective": None,
            "rule_6_preposition_score": 0,
            "rule_6_verb_score": 0,
            "rule_6_adjective_score": 0,
            "rule_6_verb_adjective_score": 0,
            "rule_6_sentiment_value": 0,
            "rule_6_sentiment_label": "Tidak terdeteksi",
        }
    )


def apply_rule_6(df, token_column="tokens_normalized"):
    result = df[token_column].apply(classify_rule_6_preposisi_verba_adjektif)
    return pd.concat([df, result], axis=1)
