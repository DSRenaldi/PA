import re

import pandas as pd


ALLOWED_PUNCTUATION = ".,?!"


def retain_allowed_symbols(text):
    if pd.isna(text):
        return ""

    text = str(text)
    return re.sub(rf"[^a-zA-Z\s{re.escape(ALLOWED_PUNCTUATION)}]", " ", text)


def apply_rule_7(series):
    return series.apply(retain_allowed_symbols)
