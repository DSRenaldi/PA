from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

from utility.data_api import SENTIMENT_ORDER


DASHBOARD_DIR = Path(__file__).resolve().parents[1]
DB_PATH = DASHBOARD_DIR / "Database" / "ABSA_insight.db"

DISPLAY_TABLES = {
    "trial_test_data": "Trial - Test Data",
    "trial_train_data": "Trial - Train Data",
    "user_data": "User Data",
}

SENTIMENT_ALIASES = {
    "positif": "Positif",
    "positive": "Positif",
    "1": "Positif",
    "1.0": "Positif",
    "netral": "Netral",
    "neutral": "Netral",
    "0": "Netral",
    "0.0": "Netral",
    "negatif": "Negatif",
    "negative": "Negatif",
    "-1": "Negatif",
    "-1.0": "Negatif",
}


def get_database_options() -> list[str]:
    return list(DISPLAY_TABLES.keys())


def get_database_option_labels() -> dict[str, str]:
    return DISPLAY_TABLES.copy()


def _validate_table_name(table_name: str) -> None:
    if table_name not in DISPLAY_TABLES:
        allowed_tables = ", ".join(DISPLAY_TABLES)
        raise ValueError(f"Table tidak tersedia: {table_name}. Pilihan: {allowed_tables}")


def get_database_mtime(db_path: Path = DB_PATH) -> float:
    return db_path.stat().st_mtime


def normalize_sentiment_label(value: object) -> str:
    normalized_value = str(value).strip().casefold()
    return SENTIMENT_ALIASES.get(normalized_value, str(value).strip())


@st.cache_data(show_spinner=False)
def get_table_data(
    table_name: str,
    db_path: Path = DB_PATH,
    db_mtime: float | None = None,
) -> pd.DataFrame:
    del db_mtime
    _validate_table_name(table_name)
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query(f"select * from {table_name}", connection)


def get_table_comment_data(table_name: str) -> pd.DataFrame:
    data = get_table_data(table_name, db_mtime=get_database_mtime()).copy()

    if "predicted_aspect" not in data.columns and "true_aspect" in data.columns:
        data = data.rename(columns={"true_aspect": "predicted_aspect"})

    required_columns = ["comment_id", "comment_text", "predicted_aspect", "final_sentiment_label"]
    for column in required_columns:
        if column not in data.columns:
            data[column] = ""

    data = data[required_columns].dropna(subset=["predicted_aspect", "final_sentiment_label"]).copy()
    data["predicted_aspect"] = data["predicted_aspect"].astype(str).str.strip()
    data["final_sentiment_label"] = data["final_sentiment_label"].apply(normalize_sentiment_label)
    data["comment_text"] = data["comment_text"].astype(str).str.strip()

    return data[
        (data["predicted_aspect"] != "")
        & (data["final_sentiment_label"].isin(SENTIMENT_ORDER))
    ].reset_index(drop=True)


def get_table_display_data(table_name: str) -> pd.DataFrame:
    data = get_table_data(table_name, db_mtime=get_database_mtime()).copy()

    if "predicted_aspect" not in data.columns and "true_aspect" in data.columns:
        data = data.rename(columns={"true_aspect": "predicted_aspect"})

    required_columns = [
        "comment_id",
        "postUrl",
        "comment_text",
        "date",
        "ownerUsername",
        "tanggal_input",
        "segmented_text",
        "predicted_aspect",
        "final_sentiment_label",
    ]
    for column in required_columns:
        if column not in data.columns:
            data[column] = ""

    display_columns = [
        "comment_id",
        "comment_text",
        "segmented_text",
        "predicted_aspect",
        "final_sentiment_label",
    ]
    if table_name == "user_data":
        display_columns = [
            "comment_id",
            "postUrl",
            "comment_text",
            "date",
            "ownerUsername",
            "tanggal_input",
            "segmented_text",
            "predicted_aspect",
            "final_sentiment_label",
        ]

    data = data[display_columns].dropna(subset=["predicted_aspect", "final_sentiment_label"]).copy()
    data["predicted_aspect"] = data["predicted_aspect"].astype(str).str.strip()
    data["final_sentiment_label"] = data["final_sentiment_label"].apply(normalize_sentiment_label)
    data["comment_text"] = data["comment_text"].astype(str).str.strip()
    data["segmented_text"] = data["segmented_text"].astype(str).str.strip()
    if table_name == "user_data":
        data = data.rename(columns={"date": "Tanggal Posting", "ownerUsername": "Username"})

    return data[
        (data["predicted_aspect"] != "")
        & (data["final_sentiment_label"].isin(SENTIMENT_ORDER))
    ].reset_index(drop=True)
