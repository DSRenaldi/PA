from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None


MODEL_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = MODEL_DIR.parent
SEMHAS_DIR = DASHBOARD_DIR.parent
CODE_DIR = SEMHAS_DIR / "Code"
DB_PATH = DASHBOARD_DIR / "Database" / "ABSA_insight.db"

DICTIONARY_DIR = MODEL_DIR / "Dictionary"
KAMUS_DIR = MODEL_DIR / "Kamus"
LOCAL_TRAIN_PATH = MODEL_DIR / "Dataset" / "segmented_dataset.csv"
FALLBACK_TRAIN_PATH = CODE_DIR / "Output_V3" / "train_data.csv"

INPUT_COLUMNS = ["postUrl", "comment_text", "ownerUsername", "date", "month"]
MODEL_OUTPUT_COLUMNS = [
    "comment_id",
    "segmented_text",
    "predicted_aspect",
    "final_sentiment_label",
]
USER_DATA_COLUMNS = [
    "comment_id",
    "postUrl",
    "comment_text",
    "ownerUsername",
    "date",
    "month",
    "tanggal_input",
    "segmented_text",
    "predicted_aspect",
    "final_sentiment_label",
]

SENTIMENT_FALLBACK = "Netral"
PRIMARY_THRESHOLD = 0.60
LAINNYA_THRESHOLD = 0.18


def _read_csv(path_or_file) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            if hasattr(path_or_file, "seek"):
                path_or_file.seek(0)
            return pd.read_csv(path_or_file, encoding=encoding)
        except UnicodeDecodeError:
            continue
    if hasattr(path_or_file, "seek"):
        path_or_file.seek(0)
    return pd.read_csv(path_or_file)


def _load_text_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    }


def _load_slang_words(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return {str(key).lower(): str(value).lower() for key, value in data.items()}


def _load_emoji_pattern(path: Path) -> re.Pattern | None:
    if not path.exists():
        return None
    try:
        emoji_df = pd.read_csv(path)
    except Exception:
        return None

    emoji_values: list[str] = []
    for column in emoji_df.columns:
        emoji_values.extend(
            str(value)
            for value in emoji_df[column].dropna().tolist()
            if str(value).strip()
        )

    if not emoji_values:
        return None
    return re.compile("|".join(re.escape(value) for value in emoji_values))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+|[.,!?]", str(text).lower())


def _ngrams(tokens: list[str], min_n: int = 1, max_n: int = 3) -> Iterable[str]:
    for n in range(min_n, max_n + 1):
        if len(tokens) < n:
            continue
        for index in range(len(tokens) - n + 1):
            yield " ".join(tokens[index : index + n])


def _get_stemmer():
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    except ImportError:
        return None
    return StemmerFactory().create_stemmer()


def _import_sentiment_class():
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))
    from sentiment import Sentiment

    return Sentiment


class AspectClassifier:
    def __init__(
        self,
        training_path: Path | None = None,
        min_df: int = 2,
        primary_threshold: float = PRIMARY_THRESHOLD,
        lainnya_threshold: float = LAINNYA_THRESHOLD,
    ):
        self.training_path = training_path or self._resolve_training_path()
        self.min_df = min_df
        self.primary_threshold = primary_threshold
        self.lainnya_threshold = lainnya_threshold
        self.idf: dict[str, float] = {}
        self.doc_labels: list[str] = []
        self.doc_norms: list[float] = []
        self.inverted_index: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.vectorizer = None
        self.train_matrix = None
        self.train_labels: list[str] = []
        self.fit()

    @staticmethod
    def _resolve_training_path() -> Path:
        if LOCAL_TRAIN_PATH.exists():
            return LOCAL_TRAIN_PATH
        return FALLBACK_TRAIN_PATH

    def fit(self) -> None:
        if not self.training_path.exists():
            raise FileNotFoundError(f"Data training aspek tidak ditemukan: {self.training_path}")

        train_df = _read_csv(self.training_path)
        text_column = self._pick_column(
            train_df,
            ["stemmed_segmented_text", "segmented_text", "comment_text"],
        )
        label_column = self._pick_column(train_df, ["true_aspect", "predicted_aspect"])

        records = (
            train_df[[text_column, label_column]]
            .dropna()
            .astype({text_column: "string", label_column: "string"})
        )
        texts = records[text_column].tolist()
        self.train_labels = records[label_column].tolist()

        if TfidfVectorizer is not None:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=self.min_df)
            self.train_matrix = self.vectorizer.fit_transform(texts)
            return

        tokenized_docs = [list(_ngrams(_tokenize(text))) for text in texts]
        labels = self.train_labels

        document_frequency = Counter()
        for terms in tokenized_docs:
            document_frequency.update(set(terms))

        total_docs = len(tokenized_docs)
        self.idf = {
            term: math.log((1 + total_docs) / (1 + freq)) + 1
            for term, freq in document_frequency.items()
            if freq >= self.min_df
        }

        self.doc_labels = []
        self.doc_norms = []
        self.inverted_index = defaultdict(list)

        for terms, label in zip(tokenized_docs, labels):
            vector = self._vector_from_terms(terms)
            norm = self._norm(vector)
            if not vector or not norm:
                continue

            doc_index = len(self.doc_labels)
            self.doc_labels.append(label)
            self.doc_norms.append(norm)
            for term, value in vector.items():
                self.inverted_index[term].append((doc_index, value))

    @staticmethod
    def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
        for column in candidates:
            if column in df.columns:
                return column
        raise ValueError(f"Kolom yang dibutuhkan tidak ditemukan. Kandidat: {candidates}")

    def _vector_from_terms(self, terms: list[str]) -> dict[str, float]:
        counts = Counter(term for term in terms if term in self.idf)
        if not counts:
            return {}
        total_terms = sum(counts.values())
        return {
            term: (count / total_terms) * self.idf[term]
            for term, count in counts.items()
        }

    @staticmethod
    def _norm(vector: dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values()))

    def _vectorize(self, text: str) -> tuple[dict[str, float], float]:
        vector = self._vector_from_terms(list(_ngrams(_tokenize(text))))
        return vector, self._norm(vector)

    def predict(self, text: str) -> str:
        if self.vectorizer is not None and self.train_matrix is not None:
            query_vector = self.vectorizer.transform([text])
            similarities = cosine_similarity(query_vector, self.train_matrix).flatten()
            best_index = int(similarities.argmax())
            best_score = float(similarities[best_index])
            best_aspect = self.train_labels[best_index]

            if best_score >= self.primary_threshold:
                return best_aspect
            if best_score < self.lainnya_threshold:
                return "Lainnya"
            return best_aspect

        vector, vector_norm = self._vectorize(text)
        if not vector or not vector_norm:
            return "Lainnya"

        candidate_scores = Counter()
        for term, value in vector.items():
            for doc_index, doc_value in self.inverted_index.get(term, []):
                candidate_scores[doc_index] += value * doc_value

        best_aspect = "Lainnya"
        best_score = 0.0
        for doc_index, dot_product in candidate_scores.items():
            denominator = vector_norm * self.doc_norms[doc_index]
            score = dot_product / denominator if denominator else 0.0
            if score > best_score:
                best_score = score
                best_aspect = self.doc_labels[doc_index]

        if best_score >= self.primary_threshold:
            return best_aspect
        if best_score < self.lainnya_threshold:
            return "Lainnya"
        return best_aspect


class UserInputModel:
    def __init__(self):
        self.stop_words = _load_text_set(DICTIONARY_DIR / "combined_stop_words.txt")
        self.slang_words = _load_slang_words(DICTIONARY_DIR / "update_combined_slang_words.json")
        self.conjunction_words = _load_text_set(DICTIONARY_DIR / "augmentation_text_dict.txt")
        self.emoji_pattern = _load_emoji_pattern(DICTIONARY_DIR / "emoji_underscore.csv")
        self.stemmer = _get_stemmer()

        Sentiment = _import_sentiment_class()
        self.sentiment = Sentiment(kamus_dir=str(KAMUS_DIR))
        self.aspect_classifier = AspectClassifier()

    def clean_text(self, text: str) -> str:
        cleaned = str(text).lower()
        if self.emoji_pattern is not None:
            cleaned = self.emoji_pattern.sub(" ", cleaned)
        cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
        cleaned = re.sub(r"[@#]\w+", " ", cleaned)
        cleaned = cleaned.replace("-", " ")
        cleaned = re.sub(r"\d+", " ", cleaned)
        cleaned = re.sub(r"[^a-zA-Z\s.,?!]", " ", cleaned)
        cleaned = re.sub(r"([.,?!])\1+", r"\1", cleaned)
        cleaned = re.sub(r"([.,?!])", r" \1 ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def formalize_text(self, text: str) -> str:
        tokens = _tokenize(text)
        normalized_tokens: list[str] = []
        punctuation = {".", ",", "!", "?"}

        for token in tokens:
            token = self.slang_words.get(token, token)
            if token in punctuation or token not in self.stop_words:
                normalized_tokens.append(token)

        return " ".join(normalized_tokens).strip()

    def split_text_to_segments(self, text: str) -> list[str]:
        if not text:
            return []
        if not self.conjunction_words:
            return [text]

        pattern = r"\b(?:" + "|".join(re.escape(word) for word in self.conjunction_words) + r")\b"
        parts = re.split(pattern, text, flags=re.IGNORECASE)
        segments = [re.sub(r"\s+", " ", part).strip(" ,") for part in parts]
        return [segment for segment in segments if len(segment.split()) > 1]

    def stem_text(self, text: str) -> str:
        if self.stemmer is None:
            return text
        return self.stemmer.stem(text)

    def predict_sentiment(self, text: str) -> str:
        try:
            prediction = self.sentiment.predict(text)
            return str(prediction.get("label", SENTIMENT_FALLBACK))
        except Exception:
            return SENTIMENT_FALLBACK

    def process_dataframe(
        self,
        df: pd.DataFrame,
        include_source_columns: bool = True,
    ) -> pd.DataFrame:
        missing_columns = [column for column in INPUT_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Kolom input CSV belum lengkap: {', '.join(missing_columns)}")

        input_df = df[INPUT_COLUMNS].copy()
        input_df = input_df[
            ~input_df["ownerUsername"]
            .astype(str)
            .str.contains("pdamsuryasembada", case=False, na=False)
        ].reset_index(drop=True)
        input_df["comment_id"] = range(1, len(input_df) + 1)

        rows: list[dict[str, object]] = []
        tanggal_input = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for source in input_df.to_dict("records"):
            cleaned_text = self.clean_text(source["comment_text"])
            normalized_text = self.formalize_text(cleaned_text)
            segments = self.split_text_to_segments(normalized_text)

            if not segments and len(normalized_text.split()) > 1:
                segments = [normalized_text]

            for segment in segments:
                stemmed_segment = self.stem_text(segment)
                model_output = {
                    "comment_id": source["comment_id"],
                    "segmented_text": segment,
                    "predicted_aspect": self.aspect_classifier.predict(stemmed_segment),
                    "final_sentiment_label": self.predict_sentiment(segment),
                }

                if include_source_columns:
                    rows.append(
                        {
                            **source,
                            "tanggal_input": tanggal_input,
                            **model_output,
                        }
                    )
                else:
                    rows.append(model_output)

        columns = USER_DATA_COLUMNS if include_source_columns else MODEL_OUTPUT_COLUMNS
        return pd.DataFrame(rows, columns=columns)

    def process_csv(
        self,
        csv_path_or_file,
        include_source_columns: bool = True,
    ) -> pd.DataFrame:
        return self.process_dataframe(
            _read_csv(csv_path_or_file),
            include_source_columns=include_source_columns,
        )


@lru_cache(maxsize=1)
def get_model() -> UserInputModel:
    return UserInputModel()


def process_user_csv(
    csv_path_or_file,
    include_source_columns: bool = False,
) -> pd.DataFrame:
    return get_model().process_csv(
        csv_path_or_file,
        include_source_columns=include_source_columns,
    )


def process_user_dataframe(
    df: pd.DataFrame,
    include_source_columns: bool = False,
) -> pd.DataFrame:
    return get_model().process_dataframe(
        df,
        include_source_columns=include_source_columns,
    )


def _ensure_user_data_columns(connection: sqlite3.Connection) -> None:
    existing_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(user_data)").fetchall()
    }
    column_types = {
        "comment_id": "INTEGER",
        "postUrl": "TEXT",
        "comment_text": "TEXT",
        "ownerUsername": "TEXT",
        "date": "TEXT",
        "month": "TEXT",
        "tanggal_input": "TEXT",
        "segmented_text": "TEXT",
        "predicted_aspect": "TEXT",
        "final_sentiment_label": "TEXT",
    }

    if not existing_columns:
        definitions = ", ".join(
            f"{column} {column_type}" for column, column_type in column_types.items()
        )
        connection.execute(f"CREATE TABLE user_data ({definitions})")
        return

    for column, column_type in column_types.items():
        if column not in existing_columns:
            connection.execute(f"ALTER TABLE user_data ADD COLUMN {column} {column_type}")


def save_to_user_data(processed_df: pd.DataFrame, replace: bool = False) -> int:
    missing_columns = [column for column in USER_DATA_COLUMNS if column not in processed_df.columns]
    if missing_columns:
        raise ValueError(
            "Data yang disimpan harus memuat kolom: " + ", ".join(missing_columns)
        )

    data_to_save = processed_df[USER_DATA_COLUMNS].copy()
    with sqlite3.connect(DB_PATH) as connection:
        _ensure_user_data_columns(connection)
        if replace:
            connection.execute("DELETE FROM user_data")
        data_to_save.to_sql("user_data", connection, if_exists="append", index=False)
        connection.commit()

    return len(data_to_save)


def process_and_save_user_csv(csv_path_or_file, replace: bool = False) -> pd.DataFrame:
    processed_df = get_model().process_csv(csv_path_or_file, include_source_columns=True)
    save_to_user_data(processed_df, replace=replace)
    return processed_df[MODEL_OUTPUT_COLUMNS].copy()


def process_and_save_user_dataframe(df: pd.DataFrame, replace: bool = False) -> pd.DataFrame:
    processed_df = get_model().process_dataframe(df, include_source_columns=True)
    save_to_user_data(processed_df, replace=replace)
    return processed_df[MODEL_OUTPUT_COLUMNS].copy()
