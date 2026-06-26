import altair as alt
import pandas as pd
import streamlit as st

from utility.data_api import (
    PDAM_LOGO_PATH,
    SENTIMENT_ORDER,
    compact_number,
    format_id_number,
    get_aspect_summary,
)
from utility.db_api import get_database_option_labels, get_database_options, get_table_comment_data

SENTIMENT_COLORS = {"Positif": "#22c55e", "Netral": "#a6c421", "Negatif": "#ef4444"}
BADGE_COLORS = {
    "Positif": "green",
    "Netral": "orange",
    "Negatif": "red",
}


def make_distribution_donut(summary: pd.DataFrame) -> alt.LayerChart:
    top_items = summary.head(10)[["Nama Aspek", "Frekuensi"]].copy()
    total = int(top_items["Frekuensi"].sum())
    top_items["Persentase"] = top_items["Frekuensi"] / total * 100 if total else 0
    color_range = [
        "#0ea5d7", "#05070a", "#ef4444", "#f59e0b", "#cbd5e1",
        "#22c55e", "#8b5cf6", "#14b8a6", "#64748b", "#f97316",
    ]

    donut = (
        alt.Chart(top_items)
        .mark_arc(innerRadius=80, outerRadius=115)
        .encode(
            theta=alt.Theta("Frekuensi:Q", stack=True),
            color=alt.Color(
                "Nama Aspek:N",
                scale=alt.Scale(range=color_range),
                legend=None,
            ),
            tooltip=["Nama Aspek", "Frekuensi", alt.Tooltip("Persentase:Q", format=".1f")],
        )
    )
    total_text = (
        alt.Chart(pd.DataFrame({"text": [compact_number(total)]}))
        .mark_text(fontSize=34, fontWeight="bold", color="#0b1c30", dy=-10)
        .encode(text="text:N")
    )
    label_text = (
        alt.Chart(pd.DataFrame({"text": ["Mentions"]}))
        .mark_text(fontSize=13, color="#334155", dy=22)
        .encode(text="text:N")
    )
    return (
        (donut + total_text + label_text)
        .properties(
            height=285,
            background="#DAE2FD" # <--- Membuat latar belakang grafik tembus pandang
        )
        .configure(
            view=alt.ViewConfig(stroke=None), # Menghilangkan border kotak chart
            axis=alt.AxisConfig(grid=False, domain=False, ticks=False) # Menghilangkan garis axis
        )
        # .properties(
        #     height=285
        # )
        # .configure_view(
        #     stroke=None
        # )
    )


def make_aspect_bar(row: pd.Series) -> alt.Chart:
    values = pd.DataFrame(
        [
            {"Sentiment": sentiment, "Jumlah": int(row[sentiment]), "Warna": SENTIMENT_COLORS[sentiment]}
            for sentiment in SENTIMENT_ORDER
        ]
    )
    return (
        alt.Chart(values)
        .mark_bar(size=10, cornerRadius=6)
        .encode(
            x=alt.X("Jumlah:Q", stack="normalize", title=None, axis=None),
            color=alt.Color("Warna:N", scale=None, legend=None),
            tooltip=["Sentiment", "Jumlah"],
        )
        .properties(
            height=28,
            background="#DAE2FD" # <--- Membuat latar belakang grafik tembus pandang
        )
        .configure(
            view=alt.ViewConfig(stroke=None), # Menghilangkan border kotak chart
            axis=alt.AxisConfig(grid=False, domain=False, ticks=False) # Menghilangkan garis axis
        )
        # .properties(height=28)
        # .configure_view(stroke=None)
    )


def render_distribution_legend(summary: pd.DataFrame) -> None:
    legend_data = summary.head(10)[["Nama Aspek", "Frekuensi"]].copy()
    total = legend_data["Frekuensi"].sum()
    for _, item in legend_data.iterrows():
        left, right = st.columns([2.4, 1])
        left.write(item["Nama Aspek"])
        right.write(f"**{item['Frekuensi'] / total:.0%}**" if total else "**0%**")


def render_aspect_card(row: pd.Series) -> None:
    with st.container(border=True):
        title_col, badge_col = st.columns([2, 1], vertical_alignment="top")
        title_col.subheader(row["Nama Aspek"])
        badge_col.badge(
            row["Most Sentiment"].upper(),
            color=BADGE_COLORS[row["Most Sentiment"]],
            width="content",
        )
        st.caption(f"{format_id_number(int(row['Frekuensi']))} Mentions")
        score = row["Avg Sentiment Score"]
        st.subheader(f"{score:.1f} / 5.0")
        st.caption("AVG SENTIMENT SCORE")
        st.altair_chart(make_aspect_bar(row), width='stretch')
        st.caption(f"POSITIF: {format_id_number(int(row['Positif']))} DATA")
        st.caption(f"NETRAL: {format_id_number(int(row['Netral']))} DATA")
        st.caption(f"NEGATIF: {format_id_number(int(row['Negatif']))} DATA")

def data_source_dropdown(table_options: list[str], table_labels: dict[str, str]) -> str:
    return st.selectbox(
        "Sumber data",
        table_options,
        format_func=lambda table_name: table_labels[table_name],
        index=0,
        key="overview_data_source",
        help="Trial - Test Data: Data test yang digunakan pada saat proses develop model.\n\n" \
        "Trial - Train Data: Data train yang digunakan pada saat proses develop model.\n\n" \
        "User Data: Data yang diinputkan oleh user",
    )


def show():
    table_options = get_database_options()
    table_labels = get_database_option_labels()

    _, page, _ = st.columns([0.08, 0.84, 0.08])

    with page:
        header_left, header_right = st.columns([3.4, 1], vertical_alignment="bottom")
        header_left.title("Aspect Analysis")
        with header_right:
            selected_table = data_source_dropdown(table_options, table_labels)

    try:
        df = get_table_comment_data(selected_table)
    except FileNotFoundError as exc:
        st.error(f"File data tidak ditemukan: {exc.filename}")
        st.stop()
    except ValueError as exc:
        st.error("Kolom data tidak sesuai.")
        st.caption(str(exc))
        st.stop()

    if df.empty:
        with page:
            st.warning(
                f"Tabel {table_labels[selected_table]} belum memiliki data aspek "
                "dan sentimen yang bisa ditampilkan."
            )
        return

    summary_df = get_aspect_summary(df)


    with page:
        with st.container(border=True):
            title_col, menu_col = st.columns([3, 0.3])
            title_col.subheader("DISTRIBUSI ASPEK")
            menu_col.write("⋮")
            chart_col, legend_col = st.columns([1.25, 1], vertical_alignment="center")
            chart_col.altair_chart(make_distribution_donut(summary_df), width='stretch')
            with legend_col:
                render_distribution_legend(summary_df)

        st.header("Analisis Per Aspek")
        st.markdown(
            "###### Ringkasan sentimen dan volume mention untuk setiap kategori keluhan "
            "###### dan layanan utama PDAM Surya Sembada."
        )

        top_aspects = summary_df.head(10)
        for start in range(0, len(top_aspects), 3):
            cols = st.columns(3)
            for col, (_, row) in zip(cols, top_aspects.iloc[start:start + 3].iterrows()):
                with col:
                    render_aspect_card(row)

        with st.container(border=True):
            st.markdown("###### MATRIKS ANALISIS MENDALAM")
            table_df = summary_df[
                ["Nama Aspek", "Frekuensi", "Most Sentiment", "Avg Sentiment Score"]
            ].copy()
            table_df["Frekuensi"] = table_df["Frekuensi"].astype(int)
            table_df["Avg Sentiment Score"] = table_df["Avg Sentiment Score"].round(2)
            st.dataframe(
                table_df,
                hide_index=True,
                width='stretch',
                column_config={
                    "Nama Aspek": st.column_config.TextColumn("NAMA ASPEK"),
                    "Frekuensi": st.column_config.NumberColumn("FREKUENSI"),
                    "Most Sentiment": st.column_config.TextColumn("MOST SENTIMENT"),
                    "Avg Sentiment Score": st.column_config.NumberColumn("AVG SCORE", format="%.2f"),
                },
            )
            st.caption(f"Menampilkan 1-{len(table_df)} dari {len(table_df)} aspek")
