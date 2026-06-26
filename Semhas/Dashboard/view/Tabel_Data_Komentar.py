import pandas as pd
import streamlit as st
import altair as alt

from utility.data_api import SENTIMENT_ORDER
from utility.db_api import get_database_option_labels, get_database_options, get_table_display_data

SENTIMENT_COLORS = {"Positif": "#22c55e", "Netral": "#a6c421", "Negatif": "#ef4444"}


def make_aspect_sentiment_chart(data: pd.DataFrame) -> alt.Chart:
    top_aspects = data["predicted_aspect"].value_counts().head(10).index.tolist()
    chart_df = (
        data[data["predicted_aspect"].isin(top_aspects)]
        .groupby(["predicted_aspect", "final_sentiment_label"])
        .size()
        .reset_index(name="Jumlah")
    )

    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("Jumlah:Q", stack="normalize", title="Persentase"),
            y=alt.Y(
                "predicted_aspect:N",
                sort=top_aspects,
                title=None,
                axis=alt.Axis(labelLimit=220),
            ),
            color=alt.Color(
                "final_sentiment_label:N",
                title="Sentimen",
                scale=alt.Scale(
                    domain=SENTIMENT_ORDER,
                    range=[SENTIMENT_COLORS[sentiment] for sentiment in SENTIMENT_ORDER],
                ),
            ),
            tooltip=["predicted_aspect", "final_sentiment_label", "Jumlah"],
        )
        .properties(height=430)
    )


def get_comment_details(data: pd.DataFrame, comment_id: int) -> pd.DataFrame:
    """Get all segments untuk comment_id tertentu."""
    return data[data["comment_id"] == comment_id].reset_index(drop=True)


def calculate_metrics(data: pd.DataFrame) -> dict:
    """Calculate metrics untuk ditampilkan di atas."""
    comment_id_counts = data["comment_id"].value_counts()
    segmented_data = int((comment_id_counts > 1).sum())
    not_segmented_data = int((comment_id_counts == 1).sum())

    return {
        "total_data": int(data["comment_id"].nunique()),
        "segmented_data": segmented_data,
        "not_segmented_data": not_segmented_data,
    }


def render_metric_card(label: str, value: str, unit: str = "") -> None:
    """Render metric card dengan border."""
    with st.container(border=True):
        st.caption(label)
        col1, col2 = st.columns([2, 1])
        col1.subheader(value)
        if unit:
            col2.caption(unit)


def render_detail_view(data: pd.DataFrame, comment_id: int) -> None:
    """Render detail view untuk comment tertentu."""
    detail_df = get_comment_details(data, comment_id)
    
    if detail_df.empty:
        st.error("Data komentar tidak ditemukan")
        return
    
    # Get unique comment text
    full_text = detail_df.iloc[0]["comment_text"]
    
    with st.container(border=True):
        st.subheader(f"KOMENTAR ASLI")
        st.markdown(f'###### *"{full_text}"*')
        if "postUrl" in detail_df.columns:
            info_cols = st.columns(4)
            info_cols[0].markdown("###### POST URL")
            info_cols[0].write(detail_df.iloc[0].get("postUrl", ""))
            info_cols[1].markdown("###### TANGGAL POSTING")
            info_cols[1].write(detail_df.iloc[0].get("Tanggal Posting", ""))
            info_cols[2].markdown("###### USERNAME")
            info_cols[2].write(detail_df.iloc[0].get("Username", ""))
            info_cols[3].markdown("###### TANGGAL INPUT")
            info_cols[3].write(detail_df.iloc[0].get("tanggal_input", ""))
    
    st.write("")
    st.subheader("SEGMENTASI ASPECT-BASED ANALYSIS")
    st.write("")
    
    # Display each segment
    for idx, (_, row) in enumerate(detail_df.iterrows(), 1):
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"###### Segment {idx}")
                st.markdown(f'*"{row["segmented_text"]}"*')
            
            with col2:
                st.markdown("###### ASPEK")
                st.write(row["predicted_aspect"])
            
            with col3:
                st.markdown("###### SENTIMEN")
                sentiment = row["final_sentiment_label"]
                color = SENTIMENT_COLORS[sentiment]
                st.markdown(
                    f'<span style="background-color: {color}; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold; font-size: 12px;">{sentiment}</span>',
                    unsafe_allow_html=True,
                )


def render_detail_popup(data: pd.DataFrame) -> None:
    selected_comment_id = st.session_state.get("selected_comment_detail_id")
    if selected_comment_id is None:
        return

    dialog = getattr(st, "dialog", None)

    def render_content() -> None:
        render_detail_view(data, int(selected_comment_id))
        if st.button("Tutup", width="stretch"):
            st.session_state.pop("selected_comment_detail_id", None)
            st.rerun()

    if dialog is None:
        with st.container(border=True):
            render_content()
        return

    @dialog(f"Detail Komentar ID: #{int(selected_comment_id):04d}", width="large")
    def detail_dialog():
        render_content()

    detail_dialog()


def show() -> None:
    table_options = get_database_options()
    table_labels = get_database_option_labels()

    _, page, _ = st.columns([0.02, 0.96, 0.02])

    with page:
        st.title("Tabel Data Komentar Lengkap")
        _, source_col, export_placeholder = st.columns([2.4, 1, 1], vertical_alignment="bottom")
        with source_col:
            selected_table = st.selectbox(
                "Sumber data",
                table_options,
                format_func=lambda table_name: table_labels[table_name],
                key="table_data_source",
            )

    try:
        df = get_table_display_data(selected_table)
    except FileNotFoundError as exc:
        st.error(f"File data tidak ditemukan: {exc.filename}")
        st.stop()
    except Exception as exc:
        st.error(f"Error loading data: {str(exc)}")
        st.stop()

    metrics = calculate_metrics(df)

    with page:
        if df.empty:
            st.warning(
                f"Tabel {table_labels[selected_table]} belum memiliki data komentar "
                "yang bisa ditampilkan."
            )
            return

        # Metric cards
        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("TOTAL DATA", f"{metrics['total_data']:,}")
        with col2:
            render_metric_card("TOTAL DATA TERSEGMENTASI", f"{metrics['segmented_data']:,}")
        with col3:
            render_metric_card("TOTAL DATA TIDAK TERSEGMENTASI", f"{metrics['not_segmented_data']:,}")

        date_range = None
        date_column = "Tanggal Posting" if "Tanggal Posting" in df.columns else None
        if date_column:
            date_values = pd.to_datetime(df[date_column], errors="coerce", dayfirst=True)
            valid_dates = date_values.dropna()
            if not valid_dates.empty:
                st.write("")
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                date_range = st.date_input(
                    "Filter Rentang Tanggal",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="table_date_range",
                )

        # Search dan filter section
        st.write("")
        search_col, aspect_col, sentiment_col = st.columns([2, 1, 1])

        with search_col:
            search_query = st.text_input(
                "🔍 Cari Keyword Komentar...",
                placeholder="Masukkan keyword...",
                label_visibility="collapsed",
                key="search_input"
            )

        with aspect_col:
            aspect_filter = st.selectbox(
                "Filter Aspek",
                ["Semua Aspek"] + sorted(df["predicted_aspect"].unique().tolist()),
                label_visibility="collapsed",
                key="aspect_filter"
            )

        with sentiment_col:
            sentiment_filter = st.selectbox(
                "Filter Sentimen",
                ["Semua Sentimen"] + SENTIMENT_ORDER,
                label_visibility="collapsed",
                key="sentiment_filter"
            )

        # Track filter state untuk reset pagination
        filter_state = f"{search_query}|{aspect_filter}|{sentiment_filter}|{date_range}"
        if "last_filter_state" not in st.session_state:
            st.session_state.last_filter_state = ""
        
        if st.session_state.last_filter_state != filter_state:
            st.session_state.data_table_page = 0
            st.session_state.last_filter_state = filter_state

        # Filter data
        filtered_df = df.copy()

        if date_column and date_range and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_dates = pd.to_datetime(
                filtered_df[date_column],
                errors="coerce",
                dayfirst=True,
            ).dt.date
            filtered_df = filtered_df[
                (filtered_dates >= start_date) & (filtered_dates <= end_date)
            ]

        if search_query:
            filtered_df = filtered_df[
                filtered_df["comment_text"].astype(str).str.contains(
                    search_query, case=False, na=False, regex=False
                )
            ]

        if aspect_filter != "Semua Aspek":
            filtered_df = filtered_df[filtered_df["predicted_aspect"] == aspect_filter]

        if sentiment_filter != "Semua Sentimen":
            filtered_df = filtered_df[filtered_df["final_sentiment_label"] == sentiment_filter]

        filtered_df = filtered_df.reset_index(drop=True)
        
        # Export button dengan filtered data
        with export_placeholder:
            st.download_button(
                label="📥 Export CSV",
                data=filtered_df.to_csv(index=False),
                file_name="komentar_data.csv",
                mime="text/csv",
                width="stretch",
            )

        # Pagination setup
        items_per_page = 10
        total_items = len(filtered_df)
        total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

        # Initialize page in session state
        if "data_table_page" not in st.session_state:
            st.session_state.data_table_page = 0

        # Ensure page is within bounds
        if st.session_state.data_table_page >= total_pages:
            st.session_state.data_table_page = 0

        # Pagination controls
        col1, col2, col3, col4, col5 = st.columns([1, 0.75, 0.45, 1.2, 1], vertical_alignment="center")
        with col1:
            if st.button("← Previous", width='stretch'):
                if st.session_state.data_table_page > 0:
                    st.session_state.data_table_page -= 1
                    st.rerun()

        with col2:
            jump_page = st.number_input(
                "Jump to page",
                min_value=1,
                max_value=total_pages,
                value=st.session_state.data_table_page + 1,
                step=1,
                key=f"jump_to_table_page_{st.session_state.data_table_page}_{total_pages}",
                label_visibility="collapsed",
            )

        with col3:
            if st.button("Go", width="stretch"):
                st.session_state.data_table_page = int(jump_page) - 1
                st.rerun()

        with col4:
            current_page = st.session_state.data_table_page + 1
            st.write(f"**Page {current_page} of {total_pages}**")

        with col5:
            if st.button("Next →", width='stretch'):
                if st.session_state.data_table_page < total_pages - 1:
                    st.session_state.data_table_page += 1
                    st.rerun()

        # Display table
        st.write("")
        start_idx = st.session_state.data_table_page * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        page_data = filtered_df.iloc[start_idx:end_idx]

        # Display table container
        with st.container(border=True):
            st.markdown(f"###### Menampilkan {start_idx + 1}-{end_idx} dari {total_items:,} entri")

            # Header row
            column_sizes = [0.5, 2.5, 2, 1.5, 1.2, 1.2]
            columns = st.columns(column_sizes)
            col1, col2, col3 = columns[0], columns[1], columns[2]
            with col1:
                st.write("**ID**")
            with col2:
                st.write("**KOMENTAR LENGKAP**")
            with col3:
                st.write("**TEKS TERSEGMEN**")
            columns[3].write("**ASPEK**")
            columns[4].write("**SENTIMEN**")
            columns[5].write("")

            st.divider()

            # Data rows
            for idx, (_, row) in enumerate(page_data.iterrows()):
                columns = st.columns(column_sizes)

                comment_text = str(row["comment_text"])
                comment_short = (
                    comment_text[:100] + "..." if len(comment_text) > 100 else comment_text
                )
                comment_segment = (
                    comment_text[:50] + "..." if len(comment_text) > 50 else comment_text
                )

                with columns[0]:
                    st.write(f"#{int(row['comment_id']):04d}")
                columns[1].write(comment_short)
                columns[2].write(f"*{comment_segment}*")
                aspect_col = columns[3]
                sentiment_col = columns[4]
                action_col = columns[5]

                with aspect_col:
                    st.write(row["predicted_aspect"])
                with sentiment_col:
                    sentiment = row["final_sentiment_label"]
                    color = SENTIMENT_COLORS[sentiment]
                    st.markdown(
                        f'<span style="background-color: {color}; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold;">{sentiment}</span>',
                        unsafe_allow_html=True,
                    )
                with action_col:
                    unique_key = f"view_{start_idx + idx}_{int(row['comment_id'])}"
                    if st.button("View", key=unique_key):
                        st.session_state["selected_comment_detail_id"] = int(row["comment_id"])
                        st.rerun()

                st.divider()
                
                # Show detail view if expanded
                if False and f"show_detail_{start_idx + idx}" in st.session_state and st.session_state[f"show_detail_{start_idx + idx}"]:
                    st.write("")
                    col_back, col_title = st.columns([1, 5])
                    with col_back:
                        if st.button("← Tutup Detail", key=f"close_{start_idx + idx}"):
                            st.session_state[f"show_detail_{start_idx + idx}"] = False
                            st.rerun()
                    
                    with col_title:
                        st.write(f"**Detail Komentar ID: #{int(row['comment_id']):04d}**")
                    
                    st.divider()
                    render_detail_view(df, int(row['comment_id']))
                    st.write("")
                    st.divider()

        render_detail_popup(df)

        # Pagination info
        st.write("")
        info_cols = st.columns([1, 3, 1])
        with info_cols[1]:
            st.markdown(f"###### Showing {start_idx + 1}-{end_idx} of {total_items:,} entries")

        st.write("")
        with st.container(border=True):
            st.subheader("Sentimen per Aspek")
            st.markdown("###### Komposisi sentimen untuk 10 aspek teratas berdasarkan data yang sedang difilter.")
            if filtered_df.empty:
                st.info("Tidak ada data untuk ditampilkan pada chart.")
            else:
                st.altair_chart(make_aspect_sentiment_chart(filtered_df), width="stretch")
