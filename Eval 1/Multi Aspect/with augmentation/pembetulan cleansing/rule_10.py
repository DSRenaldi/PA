import pandas as pd

from rule_helpers import to_label


TARGET_PHRASE = "terima kasih"


def classify_rule_10_terima_kasih_position(text):
    if pd.isna(text) or not isinstance(text, str):
        return pd.Series(
            {
                "rule_10_applied": False,
                "rule_10_phrase_position": None,
                "rule_10_sentiment_value": 0,
                "rule_10_sentiment_label": "Tidak terdeteksi",
            }
        )

    normalized_text = " ".join(text.lower().strip().split())

    if TARGET_PHRASE not in normalized_text:
        return pd.Series(
            {
                "rule_10_applied": False,
                "rule_10_phrase_position": None,
                "rule_10_sentiment_value": 0,
                "rule_10_sentiment_label": "Tidak terdeteksi",
            }
        )

    if normalized_text.startswith(TARGET_PHRASE):
        sentiment_value = 1
        phrase_position = "awal"
    else:
        sentiment_value = 0
        phrase_position = "tengah/akhir"

    return pd.Series(
        {
            "rule_10_applied": True,
            "rule_10_phrase_position": phrase_position,
            "rule_10_sentiment_value": sentiment_value,
            "rule_10_sentiment_label": to_label(sentiment_value),
        }
    )


def apply_rule_10(df, text_column="normalized_text"):
    result = df[text_column].apply(classify_rule_10_terima_kasih_position)
    return pd.concat([df, result], axis=1)
