import re

import pandas as pd

from rule_1 import classify_rule_1_verba_tunggal
from rule_2 import classify_rule_2_adjektiva_tunggal
from rule_3 import classify_rule_3_preposisi_adjektif
from rule_4 import classify_rule_4_preposisi_verba
from rule_5 import classify_rule_5_verba_adjektif
from rule_6 import classify_rule_6_preposisi_verba_adjektif
from rule_9 import merge_two_word_phrases
from rule_10 import classify_rule_10_terima_kasih_position
from rule_11 import split_valid_and_ignored_tokens
from rule_helpers import to_label


CONTRASTIVE_CONJUNCTIONS = [
    "akan tetapi",
    "tetapi",
    "tapi",
    "namun",
    "padahal",
    "sedangkan",
]

HOPE_CONJUNCTIONS = [
    "agar supaya",
    "supaya",
    "agar",
    "biar",
]

CONDITION_CONJUNCTIONS = [
    "apabila",
    "bila",
    "bilamana",
    "jika",
    "jikalau",
    "kalau",
]

REQUEST_WORDS = {"tolong", "mohon"}


def tokenize_clause(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    return re.findall(r"[a-zA-Z_]+|[.,!?]", text.lower())


def split_clauses(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    parts = re.split(r"[.,!?]+", text.lower())
    return [part.strip() for part in parts if part.strip()]


def contains_request(text):
    tokens = tokenize_clause(text)
    return any(token in REQUEST_WORDS for token in tokens)


def starts_with_phrase(text, phrase):
    normalized = " ".join(str(text).lower().split())
    return normalized.startswith(phrase)


def find_phrase_in_text(text, phrase_list):
    normalized = " ".join(str(text).lower().split())
    for phrase in phrase_list:
        if phrase in normalized:
            return phrase
    return None


def remove_phrase_from_start(text, phrase):
    normalized = " ".join(str(text).lower().split())
    if normalized.startswith(phrase):
        return normalized[len(phrase):].strip()
    return normalized


def score_clause(text):
    tokens = tokenize_clause(text)
    merged_tokens, _ = merge_two_word_phrases(tokens)
    valid_tokens, ignored_tokens = split_valid_and_ignored_tokens(merged_tokens)

    score_details = {
        "rule_1": int(classify_rule_1_verba_tunggal(valid_tokens)["rule_1_sentiment_value"]),
        "rule_2": int(classify_rule_2_adjektiva_tunggal(valid_tokens)["rule_2_sentiment_value"]),
        "rule_3": int(classify_rule_3_preposisi_adjektif(valid_tokens)["rule_3_sentiment_value"]),
        "rule_4": int(classify_rule_4_preposisi_verba(valid_tokens)["rule_4_sentiment_value"]),
        "rule_5": int(classify_rule_5_verba_adjektif(valid_tokens)["rule_5_sentiment_value"]),
        "rule_6": int(classify_rule_6_preposisi_verba_adjektif(valid_tokens)["rule_6_sentiment_value"]),
        "rule_10": int(classify_rule_10_terima_kasih_position(text)["rule_10_sentiment_value"]),
    }

    total_score = sum(score_details.values())
    normalized_score = 1 if total_score > 0 else -1 if total_score < 0 else 0

    return {
        "tokens": valid_tokens,
        "ignored_tokens": ignored_tokens,
        "score_details": score_details,
        "raw_score": total_score,
        "normalized_score": normalized_score,
        "label": to_label(normalized_score),
    }


def classify_rule_12(text):
    clauses = split_clauses(text)
    clause_scores = [score_clause(clause) for clause in clauses]

    default_result = {
        "rule_12_applied": False,
        "rule_12_type": None,
        "rule_12_subrule": None,
        "rule_12_conjunction": None,
        "rule_12_clause_count": len(clauses),
        "rule_12_clause_1": clauses[0] if len(clauses) > 0 else "",
        "rule_12_clause_2": clauses[1] if len(clauses) > 1 else "",
        "rule_12_clause_3": clauses[2] if len(clauses) > 2 else "",
        "rule_12_sentiment_value": 0,
        "rule_12_sentiment_label": "Tidak terdeteksi",
    }

    for idx, clause in enumerate(clauses):
        conjunction = find_phrase_in_text(clause, CONTRASTIVE_CONJUNCTIONS)
        if conjunction and idx > 0:
            left_score = clause_scores[idx - 1]["normalized_score"]
            right_text = remove_phrase_from_start(clause, conjunction)
            right_score = score_clause(right_text)["normalized_score"]

            if left_score == -1 or right_score == -1:
                sentiment_value = -1
            elif left_score == 1 and right_score == 1:
                sentiment_value = 1
            else:
                sentiment_value = 0

            result = default_result.copy()
            result.update(
                {
                    "rule_12_applied": True,
                    "rule_12_type": "konjungsi_koordinatif_bertentangan",
                    "rule_12_subrule": "12.1",
                    "rule_12_conjunction": conjunction,
                    "rule_12_sentiment_value": sentiment_value,
                    "rule_12_sentiment_label": to_label(sentiment_value),
                }
            )
            return pd.Series(result)

    for idx, clause in enumerate(clauses):
        conjunction = find_phrase_in_text(clause, HOPE_CONJUNCTIONS)
        if conjunction:
            previous_non_request_idx = None
            for j in range(idx - 1, -1, -1):
                if not contains_request(clauses[j]):
                    previous_non_request_idx = j
                    break

            hope_text = clause
            if starts_with_phrase(clause, conjunction):
                hope_text = remove_phrase_from_start(clause, conjunction)
            elif conjunction in clause:
                hope_text = clause.split(conjunction, 1)[1].strip()

            hope_score = score_clause(hope_text)["normalized_score"]

            if previous_non_request_idx is not None:
                sentiment_value = clause_scores[previous_non_request_idx]["normalized_score"]
                result = default_result.copy()
                result.update(
                    {
                        "rule_12_applied": True,
                        "rule_12_type": "konjungsi_subordinatif_harapan",
                        "rule_12_subrule": "12.2.1",
                        "rule_12_conjunction": conjunction,
                        "rule_12_sentiment_value": sentiment_value,
                        "rule_12_sentiment_label": to_label(sentiment_value),
                    }
                )
                return pd.Series(result)

            sentiment_value = hope_score
            result = default_result.copy()
            result.update(
                {
                    "rule_12_applied": True,
                    "rule_12_type": "konjungsi_subordinatif_harapan",
                    "rule_12_subrule": "12.2.2",
                    "rule_12_conjunction": conjunction,
                    "rule_12_sentiment_value": sentiment_value,
                    "rule_12_sentiment_label": to_label(sentiment_value),
                }
            )
            return pd.Series(result)

    for idx, clause in enumerate(clauses):
        conjunction = find_phrase_in_text(clause, CONDITION_CONJUNCTIONS)
        if conjunction and starts_with_phrase(clause, conjunction):
            if idx == 0:
                conditional_text = remove_phrase_from_start(clause, conjunction)
                sentiment_value = score_clause(conditional_text)["normalized_score"]
                result = default_result.copy()
                result.update(
                    {
                        "rule_12_applied": True,
                        "rule_12_type": "konjungsi_subordinatif_syarat",
                        "rule_12_subrule": "12.3.1",
                        "rule_12_conjunction": conjunction,
                        "rule_12_sentiment_value": sentiment_value,
                        "rule_12_sentiment_label": to_label(sentiment_value),
                    }
                )
                return pd.Series(result)

            sentiment_value = clause_scores[idx - 1]["normalized_score"]
            result = default_result.copy()
            result.update(
                {
                    "rule_12_applied": True,
                    "rule_12_type": "konjungsi_subordinatif_syarat",
                    "rule_12_subrule": "12.3.2",
                    "rule_12_conjunction": conjunction,
                    "rule_12_sentiment_value": sentiment_value,
                    "rule_12_sentiment_label": to_label(sentiment_value),
                }
            )
            return pd.Series(result)

    return pd.Series(default_result)


def apply_rule_12(df, text_column="normalized_text"):
    result = df[text_column].apply(classify_rule_12)
    return pd.concat([df, result], axis=1)
