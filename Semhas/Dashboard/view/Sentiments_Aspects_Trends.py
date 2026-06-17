from __future__ import annotations

import calendar

import altair as alt
import pandas as pd
import streamlit as st

from utility.data_api import SENTIMENT_ORDER, format_id_number
from utility.db_api import get_database_mtime, get_table_data, normalize_sentiment_label


SENTIMENT_COLORS = {
    "Positif": "#16a34a",
    "Netral": "#a6c421",
    "Negatif": "#ef4444",
}
ASPECT_ORDER = [
    "Air Kotor/Bau",
    "Air Tidak Mengalir",
    "Pelayanan",
    "Harga",
    "Kebocoran",
    "Meteran (Macet/Bermasalah)",
    "Pemakaian",
    "Pemasangan",
    "Perubahan Kode Tarif",
    "Lainnya",
]


def _load_user_data() -> pd.DataFrame:
    data = get_table_data("user_data", db_mtime=get_database_mtime()).copy()
    required_columns = ["predicted_aspect", "final_sentiment_label", "tanggal_input", "date", "month"]
    for column in required_columns:
        if column not in data.columns:
            data[column] = ""

    data["predicted_aspect"] = data["predicted_aspect"].astype(str).str.strip()
    data["final_sentiment_label"] = data["final_sentiment_label"].apply(normalize_sentiment_label)
    data["tanggal_input"] = pd.to_datetime(data["tanggal_input"], errors="coerce")
    data["date_value"] = pd.to_datetime(data["date"], errors="coerce", dayfirst=True)
    data["date_value"] = data["date_value"].fillna(data["tanggal_input"])

    data = data[
        (data["predicted_aspect"] != "")
        & (data["final_sentiment_label"].isin(SENTIMENT_ORDER))
    ].copy()

    return data.sort_values("date_value").reset_index(drop=True)


def _make_summary_cards(df: pd.DataFrame) -> None:
    total = len(df)
    counts = df["final_sentiment_label"].value_counts()
    cols = st.columns(4)

    metrics = [
        ("Total Ulasan", format_id_number(total), "#0b1c30"),
        ("Sentimen Positif", f"{counts.get('Positif', 0) / total * 100:.0f}%" if total else "0%", SENTIMENT_COLORS["Positif"]),
        ("Sentimen Netral", f"{counts.get('Netral', 0) / total * 100:.0f}%" if total else "0%", SENTIMENT_COLORS["Netral"]),
        ("Sentimen Negatif", f"{counts.get('Negatif', 0) / total * 100:.0f}%" if total else "0%", SENTIMENT_COLORS["Negatif"]),
    ]

    for col, (label, value, color) in zip(cols, metrics):
        with col.container(border=True):
            st.caption(label.upper())
            st.markdown(f"<h2 style='color:{color}; margin:0'>{value}</h2>", unsafe_allow_html=True)


def _get_month_options(df: pd.DataFrame) -> list[pd.Timestamp]:
    month_values = (
        df.dropna(subset=["date_value"])["date_value"]
        .dt.to_period("M")
        .drop_duplicates()
        .sort_values()
    )
    return [month.to_timestamp() for month in month_values]


def _filter_period_data(
    df: pd.DataFrame,
    mode: str,
    selected_month: pd.Timestamp | None,
) -> tuple[pd.DataFrame, str]:
    date_df = df.dropna(subset=["date_value"]).copy()
    if date_df.empty:
        return date_df, ""

    if mode == "Per Bulan":
        month_value = selected_month or date_df["date_value"].max().to_period("M").to_timestamp()
        filtered_df = date_df[
            date_df["date_value"].dt.to_period("M") == month_value.to_period("M")
        ].copy()
        return filtered_df, month_value.strftime("%B %Y")

    latest_year = int(date_df["date_value"].dt.year.max())
    filtered_df = date_df[date_df["date_value"].dt.year == latest_year].copy()
    return filtered_df, str(latest_year)


def _make_volume_chart(df: pd.DataFrame, mode: str, period_label: str) -> alt.Chart:
    chart_df = df.dropna(subset=["date_value"]).copy()

    if mode == "Per Bulan":
        if chart_df.empty:
            base_dates = pd.date_range("2026-01-01", periods=31, freq="D")
        else:
            first_date = chart_df["date_value"].min()
            last_day = calendar.monthrange(first_date.year, first_date.month)[1]
            base_dates = pd.date_range(
                start=pd.Timestamp(first_date.year, first_date.month, 1),
                periods=last_day,
                freq="D",
            )
        chart_df["Periode"] = chart_df["date_value"].dt.floor("D")
        full_period = pd.MultiIndex.from_product(
            [base_dates, SENTIMENT_ORDER],
            names=["Periode", "final_sentiment_label"],
        )
        label_format = "%d"
        x_title = f"Tanggal - {period_label}"
    else:
        year = int(chart_df["date_value"].dt.year.max()) if not chart_df.empty else 2026
        base_dates = pd.date_range(start=f"{year}-01-01", periods=12, freq="MS")
        chart_df["Periode"] = chart_df["date_value"].dt.to_period("M").dt.to_timestamp()
        full_period = pd.MultiIndex.from_product(
            [base_dates, SENTIMENT_ORDER],
            names=["Periode", "final_sentiment_label"],
        )
        label_format = "%b"
        x_title = f"Bulan - {period_label}"

    chart_df = (
        chart_df.groupby(["Periode", "final_sentiment_label"])
        .size()
        .reindex(full_period, fill_value=0)
        .reset_index(name="Jumlah")
    )
    chart_df["Label"] = chart_df["Periode"].dt.strftime(label_format)
    label_sort = chart_df.sort_values("Periode")["Label"].drop_duplicates().tolist()

    return (
        alt.Chart(chart_df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Label:N", title=x_title, sort=label_sort),
            y=alt.Y("Jumlah:Q", title="Jumlah ulasan"),
            color=alt.Color(
                "final_sentiment_label:N",
                title="Sentimen",
                scale=alt.Scale(domain=SENTIMENT_ORDER, range=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER]),
            ),
            tooltip=["Label", "final_sentiment_label", "Jumlah"],
        )
        .properties(height=310)
    )


def _make_aspect_trend_chart(df: pd.DataFrame, mode: str, period_label: str) -> alt.Chart:
    chart_df = df.dropna(subset=["date_value"]).copy()
    aspect_sort = (
        chart_df["predicted_aspect"]
        .value_counts()
        .reindex(ASPECT_ORDER, fill_value=0)
        .sort_values(ascending=False)
        .index.tolist()
    )

    if mode == "Per Bulan":
        first_date = chart_df["date_value"].min()
        last_day = calendar.monthrange(first_date.year, first_date.month)[1]
        base_dates = pd.date_range(
            start=pd.Timestamp(first_date.year, first_date.month, 1),
            periods=last_day,
            freq="D",
        )
        chart_df["Periode"] = chart_df["date_value"].dt.floor("D")
        label_format = "%d"
        x_title = f"Tanggal - {period_label}"
    else:
        year = int(chart_df["date_value"].dt.year.max())
        base_dates = pd.date_range(start=f"{year}-01-01", periods=12, freq="MS")
        chart_df["Periode"] = chart_df["date_value"].dt.to_period("M").dt.to_timestamp()
        label_format = "%b"
        x_title = f"Bulan - {period_label}"

    full_period = pd.MultiIndex.from_product(
        [base_dates, aspect_sort],
        names=["Periode", "predicted_aspect"],
    )
    chart_df = (
        chart_df.groupby(["Periode", "predicted_aspect"])
        .size()
        .reindex(full_period, fill_value=0)
        .reset_index(name="Jumlah")
    )
    chart_df["Label"] = chart_df["Periode"].dt.strftime(label_format)
    label_sort = chart_df.sort_values("Periode")["Label"].drop_duplicates().tolist()

    return (
        alt.Chart(chart_df)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("Label:N", title=x_title, sort=label_sort),
            y=alt.Y("Jumlah:Q", title="Jumlah aspek"),
            color=alt.Color("predicted_aspect:N", title="Aspek", sort=aspect_sort),
            tooltip=["Label", "predicted_aspect", "Jumlah"],
        )
        .properties(height=340)
    )


def _make_aspect_distribution(df: pd.DataFrame) -> alt.Chart:
    aspect_sort = (
        df["predicted_aspect"]
        .value_counts()
        .reindex(ASPECT_ORDER, fill_value=0)
        .sort_values(ascending=False)
        .index.tolist()
    )
    chart_df = (
        df["predicted_aspect"]
        .value_counts()
        .reindex(ASPECT_ORDER, fill_value=0)
        .reindex(aspect_sort)
        .rename_axis("Aspek")
        .reset_index(name="Jumlah")
    )
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadius=6)
        .encode(
            x=alt.X("Jumlah:Q", title=None),
            y=alt.Y("Aspek:N", sort=aspect_sort, title=None, axis=alt.Axis(labelLimit=220)),
            color=alt.value("#565e74"),
            tooltip=["Aspek", "Jumlah"],
        )
        .properties(height=360)
    )


def _make_aspect_sentiment_chart(df: pd.DataFrame) -> alt.Chart:
    aspect_sort = (
        df["predicted_aspect"]
        .value_counts()
        .reindex(ASPECT_ORDER, fill_value=0)
        .sort_values(ascending=False)
        .index.tolist()
    )
    chart_df = (
        df.groupby(["predicted_aspect", "final_sentiment_label"])
        .size()
        .reindex(
            pd.MultiIndex.from_product(
                [ASPECT_ORDER, SENTIMENT_ORDER],
                names=["predicted_aspect", "final_sentiment_label"],
            ),
            fill_value=0,
        )
        .reset_index(name="Jumlah")
    )

    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("Jumlah:Q", stack=True, title="Jumlah"),
            y=alt.Y(
                "predicted_aspect:N",
                sort=aspect_sort,
                title=None,
                axis=alt.Axis(labelLimit=220),
            ),
            color=alt.Color(
                "final_sentiment_label:N",
                title="Sentimen",
                scale=alt.Scale(domain=SENTIMENT_ORDER, range=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER]),
            ),
            tooltip=["predicted_aspect", "final_sentiment_label", "Jumlah"],
        )
        .properties(height=360)
    )


def _make_weekly_aspect_trend(df: pd.DataFrame) -> pd.DataFrame:
    trend_df = df.dropna(subset=["date_value"]).copy()
    if trend_df.empty:
        return pd.DataFrame(
            {
                "Aspek": ASPECT_ORDER,
                "Minggu Ini": [0] * len(ASPECT_ORDER),
                "Minggu Lalu": [0] * len(ASPECT_ORDER),
                "Perubahan": ["+0"] * len(ASPECT_ORDER),
            }
        )

    trend_df["Minggu"] = trend_df["date_value"].dt.to_period("W").dt.start_time
    weeks = sorted(trend_df["Minggu"].dropna().unique())
    current_week = weeks[-1]
    previous_week = weeks[-2] if len(weeks) > 1 else None

    current = trend_df[trend_df["Minggu"] == current_week]["predicted_aspect"].value_counts()
    previous = (
        trend_df[trend_df["Minggu"] == previous_week]["predicted_aspect"].value_counts()
        if previous_week is not None
        else pd.Series(dtype=int)
    )

    rows = []
    for aspect in ASPECT_ORDER:
        current_count = int(current.get(aspect, 0))
        previous_count = int(previous.get(aspect, 0))
        change = current_count - previous_count
        rows.append(
            {
                "Aspek": aspect,
                "Minggu Ini": int(current_count),
                "Minggu Lalu": previous_count,
                "Perubahan": f"{change:+d}",
            }
        )
    return pd.DataFrame(rows)


def _make_negative_heatmap(df: pd.DataFrame) -> alt.Chart:
    chart_df = df[
        (df["final_sentiment_label"] == "Negatif") & df["date_value"].notna()
    ].copy()
    chart_df["Hari"] = chart_df["date_value"].dt.day_name()
    chart_df["Minggu"] = "Mg" + chart_df["date_value"].dt.isocalendar().week.astype(str)
    chart_df = chart_df.groupby(["Hari", "Minggu"]).size().reset_index(name="Jumlah")

    return (
        alt.Chart(chart_df)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X("Minggu:N", title=None),
            y=alt.Y("Hari:N", title=None),
            color=alt.Color("Jumlah:Q", scale=alt.Scale(scheme="reds"), title="Negatif"),
            tooltip=["Hari", "Minggu", "Jumlah"],
        )
        .properties(height=360)
    )


def show():
    _, page, _ = st.columns([0.08, 0.84, 0.08])

    with page:
        header_left, header_right = st.columns([3, 1], vertical_alignment="bottom")
        header_left.title("Sentiments & Aspects Trends")
        mode = header_right.segmented_control(
            "Periode",
            ["Per Bulan", "Per Tahun"],
            default="Per Bulan",
            label_visibility="collapsed",
        )

        df = _load_user_data()
        if df.empty:
            st.warning("Tabel user_data belum memiliki data yang bisa ditampilkan.")
            return

        selected_month = None
        if mode == "Per Bulan":
            month_options = _get_month_options(df)
            if not month_options:
                st.warning("Tabel user_data belum memiliki tanggal yang valid.")
                return
            selected_month = st.selectbox(
                "Pilih Bulan",
                month_options,
                index=len(month_options) - 1,
                format_func=lambda month: month.strftime("%B %Y"),
            )

        period_df, period_label = _filter_period_data(df, mode, selected_month)
        if period_df.empty:
            st.warning(f"Tidak ada data untuk periode {period_label}.")
            return

        _make_summary_cards(period_df)

        with st.container(border=True):
            st.subheader("Tren Persebaran Sentimen")
            st.caption(f"Jumlah ulasan per kategori sentimen berdasarkan periode ({period_label})")
            st.altair_chart(_make_volume_chart(period_df, mode, period_label), width="stretch")

        with st.container(border=True):
            st.subheader("Tren Persebaran Aspek")
            st.caption(f"Jumlah kemunculan aspek berdasarkan periode ({period_label})")
            st.altair_chart(_make_aspect_trend_chart(period_df, mode, period_label), width="stretch")

        left_col, right_col = st.columns(2)
        with left_col.container(border=True):
            st.subheader("Distribusi Aspek")
            st.caption("Aspek yang paling banyak dibahas")
            st.altair_chart(_make_aspect_distribution(period_df), width="stretch")

        with right_col.container(border=True):
            st.subheader("Sentimen per Aspek")
            st.caption("Komposisi positif, netral, dan negatif per aspek")
            st.altair_chart(_make_aspect_sentiment_chart(period_df), width="stretch")

        trend_col, heatmap_col = st.columns(2)
        with trend_col.container(border=True):
            st.subheader("Tren Aspek Mingguan")
            st.caption("Perubahan volume aspek pada minggu terbaru")
            st.dataframe(_make_weekly_aspect_trend(period_df), hide_index=True, width="stretch")

        with heatmap_col.container(border=True):
            st.subheader("Heatmap Sentimen Negatif")
            st.caption("Intensitas keluhan negatif per hari dan minggu")
            st.altair_chart(_make_negative_heatmap(period_df), width="stretch")
