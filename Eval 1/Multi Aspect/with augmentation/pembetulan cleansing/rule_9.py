import pandas as pd


PHRASE_MAP = {
    ("terima", "kasih"): "terima_kasih",
    ("air", "mata"): "air_mata",
    ("minta", "tolong"): "minta_tolong",
}


def merge_two_word_phrases(tokens, phrase_map=None):
    if not isinstance(tokens, list):
        return [], []

    phrase_map = phrase_map or PHRASE_MAP
    merged_tokens = []
    detected_phrases = []

    i = 0
    while i < len(tokens):
        current_token = str(tokens[i]).lower().strip()
        next_token = str(tokens[i + 1]).lower().strip() if i + 1 < len(tokens) else None

        if next_token is not None and (current_token, next_token) in phrase_map:
            merged_value = phrase_map[(current_token, next_token)]
            merged_tokens.append(merged_value)
            detected_phrases.append(f"{current_token} {next_token}")
            i += 2
            continue

        merged_tokens.append(current_token)
        i += 1

    return merged_tokens, detected_phrases


def apply_rule_9(df, token_column="tokens_normalized"):
    result = df[token_column].apply(merge_two_word_phrases)

    df = df.copy()
    df["rule_9_tokens"] = result.apply(lambda item: item[0])
    df["rule_9_phrases"] = result.apply(lambda item: item[1])
    df["rule_9_text"] = df["rule_9_tokens"].apply(lambda tokens: " ".join(tokens))
    return df
