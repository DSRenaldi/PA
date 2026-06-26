import altair as alt
import pandas as pd
import streamlit as st

from utility.data_api import (
    PDAM_LOGO_PATH,
    compact_number,
    get_overview_metrics,
)
from utility.db_api import (
    get_database_mtime,
    get_database_option_labels,
    get_database_options,
    get_table_data,
    get_table_comment_data,
)


def sentiment_tone(positive_percent: float) -> str:
    if positive_percent >= 60:
        return "#16a34a"
    if positive_percent >= 35:
        return "#f5b400"
    return "#ef4444"


def prepare_aspect_records(aspects: pd.DataFrame) -> list[dict]:
    records = aspects.copy()
    records["aspek"] = records["Nama Aspek"]
    records["volume"] = records["Frekuensi"]
    records["positif"] = (records["Positif"] / records["Frekuensi"] * 100).round().astype(int)
    records["warna"] = records["positif"].apply(sentiment_tone)
    return records[["aspek", "volume", "positif", "warna"]].to_dict("records")


def make_top_aspects_chart(aspects: list[dict]) -> alt.LayerChart:
    data = pd.DataFrame(aspects[:5])
    data["label"] = data["positif"].astype(str) + "% Positif"
    sort_order = [item["aspek"] for item in aspects[:5]]

    base = alt.Chart(data).encode(
        y=alt.Y(
            "aspek:N",
            sort=sort_order,
            title=None,
            axis=alt.Axis(
                labelFontSize=11,
                labelFontWeight="bold",
                labelColor="#17263a",
                labelLimit=150,
            ),
        )
    )
    track = base.mark_bar(size=8, cornerRadius=8, color="#e7f0ff").encode(
        x=alt.X("max_value:Q", title=None, axis=None, scale=alt.Scale(domain=[0, 135]))
    ).transform_calculate(max_value="100")
    
    bar = base.mark_bar(size=8, cornerRadius=8).encode(
        x=alt.X("positif:Q", title=None, axis=None, scale=alt.Scale(domain=[0, 135])),
        color=alt.Color("warna:N", scale=None, legend=None),
    )
    
    text = base.mark_text(dx=6, align="left", baseline="middle", fontSize=10, fontWeight="bold").encode(
        x=alt.X("max_text:Q", title=None, axis=None, scale=alt.Scale(domain=[0, 135])),
        text="label:N",
        color=alt.Color("warna:N", scale=None, legend=None),
    ).transform_calculate(max_text="102")

    # PERBAIKAN: Menambahkan properti background dan menyatukan konfigurasi
    return (
        (track + bar + text)
        .properties(
            height=185,
            background="#DAE2FD" # <--- Membuat latar belakang grafik tembus pandang
        )
        .configure(
            view=alt.ViewConfig(stroke=None), # Menghilangkan border kotak chart
            axis=alt.AxisConfig(grid=False, domain=False, ticks=False) # Menghilangkan garis axis
        )
    )

def make_sentiment_chart(sentiments: list[dict], total_label: str) -> alt.LayerChart:
    data = pd.DataFrame(sentiments)
    donut = (
        alt.Chart(data)
        .mark_arc(innerRadius=48, outerRadius=74)
        .encode(
            theta=alt.Theta("persen:Q", stack=True),
            color=alt.Color("warna:N", scale=None, legend=None),
            tooltip=["sentiment", "jumlah", "persen"],
        )
    )
    center_total = (
        alt.Chart(pd.DataFrame({"text": [total_label]}))
        .mark_text(fontSize=18, fontWeight="bold", color="#17263a", dy=-7)
        .encode(text="text:N")
    )
    center_label = (
        alt.Chart(pd.DataFrame({"text": ["TOTAL"]}))
        .mark_text(fontSize=9, fontWeight="bold", color="#535862", dy=13)
        .encode(text="text:N")
    )

    return (
        (donut + center_total + center_label)
        .properties(
            height=235,
            background="#DAE2FD" # <--- Membuat latar belakang grafik tembus pandang
        )
        .configure(
            view=alt.ViewConfig(stroke=None), # Menghilangkan border kotak chart
            axis=alt.AxisConfig(grid=False, domain=False, ticks=False) # Menghilangkan garis axis
        )
        # .properties(
        #     height=230
        #     )
        # .configure_view(
        #     stroke=None
        #     )
        )


def make_distribution_chart(aspects: list[dict]) -> alt.LayerChart:
    data = pd.DataFrame(aspects)
    max_volume = int(data["volume"].max())
    sort_order = [item["aspek"] for item in aspects]

    base = alt.Chart(data).encode(
        y=alt.Y(
            "aspek:N",
            sort=sort_order,
            title=None,
            axis=alt.Axis(labelFontSize=11, labelColor="#344054"),
        )
    )
    track = base.mark_bar(size=14, cornerRadius=8, color="#e7f0ff").encode(
        x=alt.X(
            "max_volume:Q",
            title=None,
            axis=None,
            scale=alt.Scale(domain=[0, max_volume + 75]),
        )
    ).transform_calculate(max_volume=str(max_volume))
    bar = base.mark_bar(size=14, cornerRadius=8, color="#55647a").encode(
        x=alt.X(
            "volume:Q",
            title=None,
            axis=None,
            scale=alt.Scale(domain=[0, max_volume + 75]),
        )
    )
    labels = base.mark_text(
        dx=10,
        align="left",
        baseline="middle",
        fontSize=11,
        fontWeight="bold",
        color="#233247",
    ).encode(
        x=alt.X(
            "label_x:Q",
            title=None,
            axis=None,
            scale=alt.Scale(domain=[0, max_volume + 75]),
        ),
        text="volume:Q",
    ).transform_calculate(label_x=str(max_volume + 20))

    return (
        (track + bar + labels)
        .properties(
            height=292,
            background="#DAE2FD" # <--- Membuat latar belakang grafik tembus pandang
        )
        .configure(
            view=alt.ViewConfig(stroke=None), # Menghilangkan border kotak chart
            axis=alt.AxisConfig(grid=False, domain=False, ticks=False) # Menghilangkan garis axis
        )
        # .properties(height=292)
        # .configure_view(stroke=None)
        # .configure_axis(grid=False, domain=False, ticks=False)
    )


def kpi(label: str, value: str) -> None:
    st.caption(label)
    st.subheader(value)


def sentiment_chip(label: str, value: str) -> None:
    with st.container(border=True):
        st.write(label)
        st.subheader(value)


def vertical_separator() -> None:
    st.write("")
    st.write("|")


def render_sentiment_legend(sentiments: list[dict]) -> None:
    for item in sentiments:
        st.write(f"**{item['sentiment_raw']}** {item['jumlah']:,} ({item['persen']}%)")


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


def apply_user_data_date_filter(df: pd.DataFrame, page) -> pd.DataFrame:
    raw_user_data = get_table_data("user_data", db_mtime=get_database_mtime())
    if "date" not in raw_user_data.columns or "comment_id" not in raw_user_data.columns:
        return df

    date_reference = raw_user_data[["comment_id", "date"]].drop_duplicates("comment_id").copy()
    date_reference["Tanggal Posting"] = pd.to_datetime(
        date_reference["date"],
        errors="coerce",
        dayfirst=True,
    )
    df = df.merge(
        date_reference[["comment_id", "Tanggal Posting"]],
        on="comment_id",
        how="left",
    )
    valid_dates = df["Tanggal Posting"].dropna()
    if valid_dates.empty:
        return df

    with page:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()
        date_range = st.date_input(
            "Filter Rentang Tanggal",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="overview_user_date_range",
        )

    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        df = df[
            df["Tanggal Posting"].dt.date.between(start_date, end_date)
        ].copy()

    return df.drop(columns=["Tanggal Posting"])


def show() -> None:
    table_options = get_database_options()
    table_labels = get_database_option_labels()

    _, page, _ = st.columns([0.02, 0.96, 0.02])

    with page:
        header_left, header_right = st.columns([3.4, 1], vertical_alignment="bottom")
        header_left.title("Analisis Layanan Air")
        header_left.markdown("###### Pantauan sentimen pelanggan dan 10 kategori aspek.")
        with header_right:
            selected_table = data_source_dropdown(table_options, table_labels)

    try:
        df = get_table_comment_data(selected_table)
    except FileNotFoundError as exc:
        st.error(f"File data tidak ditemukan: {exc.filename}")
        st.stop()
    except ValueError as exc:
        st.error("Data tidak sesuai.")
        st.caption(str(exc))
        st.stop()

    if df.empty:
        with page:
            st.warning(
                f"Tabel {table_labels[selected_table]} belum memiliki data aspek "
                "dan sentimen yang bisa ditampilkan."
            )
        return

    if selected_table == "user_data":
        df = apply_user_data_date_filter(df, page)
        if df.empty:
            with page:
                st.warning("Tidak ada data user_data pada rentang tanggal yang dipilih.")
            return

    metrics = get_overview_metrics(df)
    total_comments = metrics["total_comments"]
    sentiments = metrics["sentiments"]
    all_aspects = prepare_aspect_records(metrics["all_aspects"])
    top_aspects = prepare_aspect_records(metrics["top_aspects"])
    top_aspect_names = {item["aspek"] for item in top_aspects}
    other_aspects = [item["aspek"] for item in all_aspects if item["aspek"] not in top_aspect_names]

    with page:
        with st.container(border=True, gap="xxlarge"):
            (
                total_col,
                sep_left,
                positive_col,
                neutral_col,
                negative_col,
                sep_right,
                top_col,
            ) = st.columns(
                [0.7, 0.1, 1, 1, 1, 0.2, 0.7],
                vertical_alignment="center",
            )
            with total_col:
                kpi("TOTAL COMMENT", f"{total_comments:,}")
            with sep_left:
                vertical_separator()
            with positive_col:
                sentiment_chip("Positif", f"{metrics['positive_count']:,}")
            with neutral_col:
                sentiment_chip("Netral", f"{metrics['neutral_count']:,}")
            with negative_col:
                sentiment_chip("Negatif", f"{metrics['negative_count']:,}")
            with sep_right:
                vertical_separator()
            with top_col:
                st.caption("TOP ASPECT")
                st.write(f"**{metrics['top_aspect'].upper()}**")

        source_col, aspect_col, sentiment_col = st.columns([0.82, 1.7, 1.35])

        with source_col.container(border=True):
            logo_area, source_area = st.columns([1.7, 1])
            logo_area.image(str(PDAM_LOGO_PATH), width=165)
            source_area.markdown("###### Sumber Data")
            st.markdown("###### INSTAGRAM")
            st.subheader("[@pdamsuryasembada](https://www.instagram.com/pdamsuryasembada/)")

        with aspect_col.container(border=True):
            title_col, badge_col = st.columns([1.7, 1])
            title_col.subheader("Aspek Layanan")
            badge_col.markdown("###### TOP 5 DARI 10")
            st.altair_chart(make_top_aspects_chart(top_aspects), width='stretch')
            st.markdown(f"###### Kategori lainnya: {', '.join(other_aspects[:5])}.")
            if st.button("Lihat Semua 10 Aspek", width='stretch'):
                st.session_state.page = "Analisis Aspek"
                st.rerun()

        with sentiment_col.container(border=True, height="stretch"):
            st.subheader("Sentiment Analysis")
            donut_col, legend_col = st.columns([1, 1])
            donut_col.altair_chart(
                make_sentiment_chart(sentiments, compact_number(total_comments)),
                width='stretch',
            )
            with legend_col:
                st.write("")
                st.write("")
                render_sentiment_legend(sentiments)

        with st.container(border=True):
            title_col, volume_col = st.columns([3, 1])
            title_col.subheader("Distribusi Aspek")
            volume_col.caption(f"Total Volume: {total_comments:,} Laporan")
            st.altair_chart(make_distribution_chart(all_aspects), width='stretch')
