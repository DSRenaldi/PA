from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Event

import pandas as pd
import streamlit as st

from Model.model import (
    INPUT_COLUMNS,
    MODEL_OUTPUT_COLUMNS,
    USER_DATA_COLUMNS,
    get_model,
    save_to_user_data,
)
from utility.db_api import DB_PATH, get_table_data


def _record_input_history(
    tanggal_input: str,
    total_data: int,
    total_segmented_data: int,
    total_not_segmented_data: int,
    total_pdam_comment_drop: int,
) -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            insert into input_history (
                tanggal_input,
                total_data,
                total_segmented_data,
                total_not_segmented_data,
                total_pdam_comment_drop
            )
            values (?, ?, ?, ?, ?)
            """,
            (
                tanggal_input,
                total_data,
                total_segmented_data,
                total_not_segmented_data,
                total_pdam_comment_drop,
            ),
        )
        connection.commit()


def _get_input_history() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(
            """
            select
                tanggal_input,
                total_data,
                total_segmented_data,
                total_not_segmented_data,
                total_pdam_comment_drop
            from input_history
            order by tanggal_input desc
            """,
            connection,
        )


def _get_last_user_comment_id() -> int:
    with sqlite3.connect(DB_PATH) as connection:
        result = connection.execute(
            "select coalesce(max(comment_id), 0) from user_data"
        ).fetchone()
    return int(result[0] or 0)


def _sample_csv() -> bytes:
    sample = pd.DataFrame(
        [
            {
                "postUrl": "https://instagram.com/p/example",
                "comment_text": "Air tidak mengalir sejak pagi.",
                "ownerUsername": "contoh_user",
                "date": "2026-06-07",
                "month": "June",
            }
        ],
        columns=INPUT_COLUMNS,
    )
    return sample.to_csv(index=False).encode("utf-8")


def _show_manual_result(result_df: pd.DataFrame) -> None:
    st.subheader("Hasil Testing Manual")
    if result_df.empty:
        st.info("Komentar tidak menghasilkan segmentasi yang dapat dianalisis.")
        return
    st.dataframe(result_df[MODEL_OUTPUT_COLUMNS], width="stretch", hide_index=True)


def _process_csv_dataframe(raw_df: pd.DataFrame, cancel_event: Event) -> str:
    tanggal_input = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_comment_id = _get_last_user_comment_id()
    total_pdam_comment_drop = int(
        raw_df["ownerUsername"]
        .astype(str)
        .str.contains("pdamsuryasembada", case=False, na=False)
        .sum()
    )
    processed_parts: list[pd.DataFrame] = []

    for index, row in raw_df.iterrows():
        if cancel_event.is_set():
            raise RuntimeError("Proses dibatalkan oleh user.")
        row_df = pd.DataFrame([row], columns=raw_df.columns)
        processed_row = get_model().process_dataframe(row_df, include_source_columns=True)
        if not processed_row.empty:
            processed_row["comment_id"] = last_comment_id + index + 1
        processed_parts.append(processed_row)

    if cancel_event.is_set():
        raise RuntimeError("Proses dibatalkan oleh user.")

    processed_df = (
        pd.concat(processed_parts, ignore_index=True)
        if processed_parts
        else pd.DataFrame(columns=USER_DATA_COLUMNS)
    )
    processed_df["tanggal_input"] = tanggal_input

    save_to_user_data(processed_df)
    comment_id_counts = processed_df["comment_id"].value_counts() if not processed_df.empty else pd.Series(dtype=int)
    segmented_comment_count = int((comment_id_counts > 1).sum())
    not_segmented_comment_count = int((comment_id_counts == 1).sum())
    _record_input_history(
        tanggal_input=tanggal_input,
        total_data=len(raw_df),
        total_segmented_data=segmented_comment_count,
        total_not_segmented_data=not_segmented_comment_count,
        total_pdam_comment_drop=total_pdam_comment_drop,
    )
    get_table_data.clear()
    return f"CSV berhasil diproses. {len(processed_df)} segment data masuk ke user_data."


def _start_csv_processing(uploaded_file) -> None:
    raw_df = pd.read_csv(uploaded_file)
    cancel_event = Event()
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_process_csv_dataframe, raw_df, cancel_event)

    st.session_state["csv_process_future"] = future
    st.session_state["csv_process_executor"] = executor
    st.session_state["csv_process_cancel_event"] = cancel_event


def _clear_csv_processing_state() -> None:
    executor = st.session_state.pop("csv_process_executor", None)
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)
    st.session_state.pop("csv_process_future", None)
    st.session_state.pop("csv_process_cancel_event", None)


def _processing_popup() -> None:
    dialog = getattr(st, "dialog", None)

    def render_content() -> None:
        future = st.session_state.get("csv_process_future")
        cancel_event = st.session_state.get("csv_process_cancel_event")

        if future is None:
            return

        if future.done():
            try:
                st.session_state["csv_process_message"] = future.result()
            except Exception as error:
                st.session_state["csv_process_error"] = str(error)
            _clear_csv_processing_state()
            st.rerun()
            return

        st.write("Silahkan tunggu, model sedang memproses data.")
        st.info("Jangan tutup halaman sampai proses selesai.")

        if st.button("Cancel"):
            if cancel_event is not None:
                cancel_event.set()
            st.warning("Permintaan cancel dikirim. Data tidak akan disimpan jika proses berhasil dihentikan.")
            st.rerun()

        if st.button("Refresh Status"):
            st.rerun()

    if dialog is None:
        render_content()
        return

    @dialog("Silahkan tunggu")
    def wait_dialog():
        render_content()

    wait_dialog()


def show():
    st.title("Input Data")
    st.caption(
        "Populate your dashboard with new insights. Upload CSV untuk menyimpan hasil ke database, "
        "atau uji satu komentar secara manual tanpa menyimpan data."
    )

    csv_col, manual_col = st.columns([1.3, 1], gap="large")

    with csv_col:
        st.subheader("CSV Dataset Upload")
        uploaded_file = st.file_uploader(
            "Upload file CSV",
            type=["csv"],
            help="Kolom wajib: postUrl, comment_text, ownerUsername, date, month.",
        )
        st.download_button(
            "Download Sample CSV",
            data=_sample_csv(),
            file_name="sample_input_data.csv",
            mime="text/csv",
        )

        st.subheader("Data Requirements")
        st.write("CSV wajib memiliki kolom berikut:")
        st.dataframe(pd.DataFrame({"": INPUT_COLUMNS}), hide_index=True, width="stretch")

        if "csv_process_message" in st.session_state:
            st.success(st.session_state.pop("csv_process_message"))
        if "csv_process_error" in st.session_state:
            st.error(st.session_state.pop("csv_process_error"))

        if st.button("Process CSV", type="primary", disabled=uploaded_file is None):
            try:
                _start_csv_processing(uploaded_file)
                st.rerun()
            except Exception as error:
                st.error(f"CSV gagal diproses: {error}")

        if "csv_process_future" in st.session_state:
            _processing_popup()

    with manual_col:
        st.subheader("Manual Entry")
        manual_text = st.text_area(
            "Manual entry berfungsi untuk testing satu komentar tanpa menyimpan hasil ke database. Hasil analisis akan ditampilkan di bawah setelah tombol ditekan.",
            placeholder="Tulis komentar yang ingin diuji di sini...",
            height=260,
        )

        if st.button("Analyze Submission", type="primary"):
            if not manual_text.strip():
                st.warning("Masukkan komentar terlebih dahulu.")
            else:
                manual_df = pd.DataFrame(
                    [
                        {
                            "postUrl": "",
                            "comment_text": manual_text,
                            "ownerUsername": "manual_input",
                            "date": "",
                            "month": "",
                        }
                    ],
                    columns=INPUT_COLUMNS,
                )
                try:
                    result_df = get_model().process_dataframe(
                        manual_df,
                        include_source_columns=False,
                    )
                    st.session_state["manual_input_result"] = result_df
                except Exception as error:
                    st.error(f"Input manual gagal diproses: {error}")

    if "manual_input_result" in st.session_state:
        _show_manual_result(st.session_state["manual_input_result"])

    st.subheader("Input History")
    history_df = _get_input_history()
    st.dataframe(history_df, width="stretch", hide_index=True)
