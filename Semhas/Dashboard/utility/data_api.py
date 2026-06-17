from pathlib import Path

import pandas as pd
import streamlit as st


DASHBOARD_DIR = Path(__file__).resolve().parents[1]
SEMHAS_DIR = DASHBOARD_DIR.parent
DATA_PATH = SEMHAS_DIR / "Code" / "Output_V2" / "result_sentiment_analysis.csv"
PDAM_LOGO_PATH = DASHBOARD_DIR / "Asset" / "pdam-surabaya.png"

REQUIRED_COLUMNS = ["comment_id", "comment_text", "predicted_aspect", "final_sentiment_label"]
SENTIMENT_ORDER = ["Positif", "Netral", "Negatif"]
SENTIMENT_LABELS = {"Positif": "Positive", "Netral": "Neutral", "Negatif": "Negative"}
SENTIMENT_COLORS = {"Positif": "#22c55e", "Netral": "#3b82f6", "Negatif": "#ef4444"}
SCORE_MAP = {"Positif": 5, "Netral": 3, "Negatif": 1}


@st.cache_data(show_spinner=False)
def get_comment_data(path: Path = DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path, usecols=REQUIRED_COLUMNS)
    data = data.dropna(subset=REQUIRED_COLUMNS).copy()
    data["predicted_aspect"] = data["predicted_aspect"].astype(str).str.strip()
    data["final_sentiment_label"] = data["final_sentiment_label"].astype(str).str.strip()
    data["comment_text"] = data["comment_text"].astype(str).str.strip()
    return data[
        (data["predicted_aspect"] != "")
        & (data["final_sentiment_label"].isin(SENTIMENT_ORDER))
    ].reset_index(drop=True)


def get_sentiment_summary(data: pd.DataFrame | None = None) -> list[dict]:
    if data is None:
        data = get_comment_data()

    total = len(data)
    counts = data["final_sentiment_label"].value_counts()
    summary = []
    for sentiment in SENTIMENT_ORDER:
        count = int(counts.get(sentiment, 0))
        percent = (count / total * 100) if total else 0
        summary.append(
            {
                "sentiment": SENTIMENT_LABELS[sentiment],
                "sentiment_raw": sentiment,
                "jumlah": count,
                "persen": round(percent, 1),
                "warna": SENTIMENT_COLORS[sentiment],
            }
        )
    return summary


def get_aspect_summary(data: pd.DataFrame | None = None) -> pd.DataFrame:
    if data is None:
        data = get_comment_data()

    counts = (
        data.groupby(["predicted_aspect", "final_sentiment_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SENTIMENT_ORDER, fill_value=0)
    )
    summary = counts.reset_index().rename(columns={"predicted_aspect": "Nama Aspek"})
    summary["Frekuensi"] = summary[SENTIMENT_ORDER].sum(axis=1)
    summary["Avg Sentiment Score"] = (
        summary["Positif"] * SCORE_MAP["Positif"]
        + summary["Netral"] * SCORE_MAP["Netral"]
        + summary["Negatif"] * SCORE_MAP["Negatif"]
    ) / summary["Frekuensi"]
    summary["Most Sentiment"] = summary[SENTIMENT_ORDER].idxmax(axis=1)
    return summary.sort_values("Frekuensi", ascending=False).reset_index(drop=True)


def get_overview_metrics(data: pd.DataFrame | None = None) -> dict:
    if data is None:
        data = get_comment_data()

    sentiments = get_sentiment_summary(data)
    aspects = get_aspect_summary(data)
    specific_aspects = aspects[aspects["Nama Aspek"].str.casefold() != "lainnya"]
    top_aspect = specific_aspects.iloc[0]["Nama Aspek"] if not specific_aspects.empty else "-"

    sentiment_counts = {
        item["sentiment_raw"]: item["jumlah"]
        for item in sentiments
    }
    return {
        "total_comments": len(data),
        "positive_count": sentiment_counts.get("Positif", 0),
        "neutral_count": sentiment_counts.get("Netral", 0),
        "negative_count": sentiment_counts.get("Negatif", 0),
        "top_aspect": top_aspect,
        "sentiments": sentiments,
        "all_aspects": aspects,
        "top_aspects": specific_aspects.head(5) if not specific_aspects.empty else aspects.head(5),
    }


def compact_number(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def format_id_number(value: int) -> str:
    return f"{value:,}".replace(",", ".")
