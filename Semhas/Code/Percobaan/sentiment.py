import os
import re

import pandas as pd


class Sentiment:
    """
    Rule-based sentiment analyzer berbasis kamus POS + polarity CSV.
    Cocok dipakai langsung dari notebook melalui kolom teks yang sudah dinormalisasi
    atau di-stemming.

    Rule 1  = Adjektiva Tunggal
    Rule 2  = Verba Tunggal
    Rule 3  = Preposisi + Adjektiva
    Rule 4  = Preposisi + Verba
    Rule 5  = Verba + Adjektiva
    Rule 6  = Preposisi + Verba + Adjektiva
    Rule 7  = Frasa dari kamus frasa
    Rule 8  = Frasa negatif kuat
    Rule 9  = Komentar berbentuk pertanyaan
    Rule 10 = Complaint context penalty

    Catatan:
    - Rule 6 pada implementasi ini lebih sering terwakili lewat kombinasi
      deteksi frasa negatif kuat dan adjektiva tunggal.
    - Satu kalimat bisa memicu lebih dari satu rule saat skor dijumlahkan.
    """

    def __init__(self, kamus_dir=None, sanitisinglist=None):
        self.this_folder = os.path.dirname(os.path.abspath(__file__))
        self.kamus_dir = kamus_dir or os.path.join(self.this_folder, "kamus")
        self.pattern = [
            "Preposisi",
            "Verba",
            "Adjektiva",
            "Nomina",
            "Keterangan",
            "Adverbia",
        ]
        self.sanitisinglist = sanitisinglist or ["se", "begitu"]
        self.negation_words = {"tidak", "tak", "bukan", "belum", "ga", "gak", "nggak", "ngga"}
        self.question_words = {
            "apa",
            "apakah",
            "bagaimana",
            "gimana",
            "kenapa",
            "kapan",
            "siapa",
            "berapa",
            "kok",
            "kah",
            "info",
            "informasi",
        }
        self.strong_negative_phrases = {
            "tidak ada respon",
            "tidak ada tanggapan",
            "tidak ada tindak lanjut",
            "air tidak mengalir",
            "air mati",
            "air tidak keluar",
            "air ga keluar",
            "keluar kecil",
            "air keluar kecil",
            "tagihan melonjak",
            "tagih naik",
            "melonjak sekali",
            "naik drastis",
            "meter rusak",
            "meter mati",
            "meter tidak jalan",
            "sulit dibaca",
            "tidak keluar",
            "belum ada respon",
            "belum ada tanggapan",
            "belum ada tindak lanjut",
            "sampai sekarang belum",
            "sampai sekarang tidak",
            "ga ada air",
            "tidak ada air",
            "air keruh",
            "ada kebocoran",
            "kebocoran pipa",
        }
        self.strong_negative_words = {
            "mati",
            "kotor",
            "keruh",
            "rusak",
            "bocor",
            "macet",
            "parah",
            "lonjak",
            "melonjak",
            "drastis",
            "lambat",
            "telat",
            "susah",
            "buntu",
            "buruk",
            "mancur",
            "tagihan",
            "melonjak",
        }
        self.complaint_words = {
            "keluh",
            "komplain",
            "adu",
            "pengaduan",
            "lapor",
            "mohon",
            "tolong",
            "cek",
            "tindak",
            "lanjut",
        }
        self.posdf = self._load_lexicon()
        self.word_to_pos = dict(zip(self.posdf["word"], self.posdf["pos"]))
        self.word_to_sentiment = dict(zip(self.posdf["word"], self.posdf["sentiment"]))
        self.frasadf = self._load_phrase_lexicon()
        self.frasa_lookup = dict(zip(self.frasadf["word"], self.frasadf["sentiment"]))
        self.max_phrase_len = (
            int(self.frasadf["word"].str.split().str.len().max()) if not self.frasadf.empty else 0
        )

    def _load_lexicon(self):
        lexicon_files = [
            "Adjektiva.csv",
            "Adverbia.csv",
            "Konjungsi.csv",
            "Keterangan.csv",
            "Nomina.csv",
            "Preposisi.csv",
            "Verba.csv",
        ]

        frames = []
        for file_name in lexicon_files:
            file_path = os.path.join(self.kamus_dir, file_name)
            if not os.path.exists(file_path):
                continue

            frame = pd.read_csv(file_path)
            required_columns = {"word", "pos", "sentiment"}
            missing_columns = required_columns.difference(frame.columns)
            if missing_columns:
                raise ValueError(
                    f"Kolom kamus tidak lengkap pada {file_name}: {sorted(missing_columns)}"
                )

            frame = frame[["word", "pos", "sentiment"]].copy()
            frame["word"] = frame["word"].astype(str).str.strip().str.lower()
            frame["pos"] = frame["pos"].astype(str).str.strip()
            frame["sentiment"] = (
                pd.to_numeric(frame["sentiment"], errors="coerce").fillna(0).astype(int)
            )
            frames.append(frame)

        if not frames:
            raise FileNotFoundError(
                f"Tidak ada file kamus yang berhasil dibaca dari folder: {self.kamus_dir}"
            )

        lexicon = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["word"], keep="first")
            .reset_index(drop=True)
        )
        return lexicon

    def _load_phrase_lexicon(self):
        phrase_candidates = [
            os.path.join(self.kamus_dir, "frasa.csv"),
            os.path.join(self.kamus_dir, "Frasa.csv"),
            os.path.join(self.this_folder, "frasa.csv"),
            os.path.join(self.this_folder, "Frasa.csv"),
        ]

        for file_path in phrase_candidates:
            if os.path.exists(file_path):
                frame = pd.read_csv(file_path)
                if {"word", "sentiment"}.issubset(frame.columns):
                    frame = frame[["word", "sentiment"]].copy()
                    frame["word"] = frame["word"].astype(str).str.strip().str.lower()
                    frame["sentiment"] = (
                        pd.to_numeric(frame["sentiment"], errors="coerce")
                        .fillna(0)
                        .astype(int)
                    )
                    return frame

        return pd.DataFrame(columns=["word", "sentiment"])

    def convertSentence(self, sentence):
        """
        Return POS tag untuk setiap kata dalam sentence.
        """
        pos = []
        for word in sentence.split():
            pos.append(self.word_to_pos.get(word, "Adverbia" if word in self.negation_words else "Unknown"))
        return pos

    def sentimentNANDOperator(self, a, b):
        if a == 0 or b == 0:
            return a + b
        return 1 if a + b > 0 else -1

    def sentimentANDOperator(self, a, b):
        if a == 0 or b == 0:
            return a + b
        return 1 if a == b else -1

    def singleRule(self, word):
        # Rule 1 / Rule 2:
        # mengambil skor dasar untuk kata tunggal yang tidak masuk kombinasi lain.
        return self.word_to_sentiment.get(word, 0)

    def verbAdjectiveRule(self, idx, words, pos):
        # Rule 5:
        # mencari pasangan Verba + Adjektiva yang akan dihitung dengan NAND operator.
        try:
            idxAdj = pos[idx + 1 :].index("Adjektiva") + (idx + 1)
            if idx + 1 <= idxAdj < idx + 3:
                adjsenti = self.singleRule(words[idxAdj])
                return idxAdj, adjsenti
            return False
        except ValueError:
            return False

    def verbRule(self, idx, words, pos):
        # Rule 2 / Rule 5:
        # jika verba diikuti adjektiva gunakan Rule 5, jika tidak maka jatuh ke Rule 2.
        verbsenti = self.singleRule(words[idx])
        try:
            isverbplusadjective, adjsenti = self.verbAdjectiveRule(idx, words, pos)
            return isverbplusadjective, self.sentimentNANDOperator(verbsenti, adjsenti)
        except TypeError:
            return idx, verbsenti

    def prepositionAdjectiveRule(self, idx, preposenti, datalist):
        # Rule 3:
        # kombinasi Preposisi + Adjektiva dihitung dengan AND operator.
        adjsenti = self.singleRule(datalist["words"][idx + 1])
        return self.sentimentANDOperator(preposenti, adjsenti)

    def prepositionVerbRule(self, idx, preposenti, datalist):
        # Rule 4 / Rule 6:
        # jika setelah preposisi ada verba saja maka Rule 4,
        # jika verba itu punya pasangan adjektiva maka mengarah ke Rule 6.
        isanyadjidx, verbsenti = self.verbRule(idx + 1, datalist["words"], datalist["pos"])
        if isanyadjidx not in (None, False):
            return isanyadjidx, self.sentimentANDOperator(preposenti, verbsenti)
        return idx, self.sentimentNANDOperator(preposenti, verbsenti)

    def prepositionRule(self, idx, words, pos):
        # Rule 3 / Rule 4 / Rule 6:
        # dispatcher untuk pola yang diawali preposisi.
        datalist = {"words": words, "pos": pos}
        preposenti = self.singleRule(words[idx])
        try:
            if pos[idx + 1] == "Adjektiva":
                return [idx + 1], self.prepositionAdjectiveRule(idx, preposenti, datalist)
            if pos[idx + 1] == "Verba":
                idxAdj, sentiment = self.prepositionVerbRule(idx, preposenti, datalist)
                return [idx + 1, idxAdj], sentiment
            return idx, 0
        except IndexError:
            return idx, 0

    def getWordSentimentValue(self, idx, words, pos):
        # Rule router:
        # Verba      -> Rule 2 atau Rule 5
        # Preposisi  -> Rule 3, Rule 4, atau Rule 6
        # selain itu -> Rule 1 jika kata punya skor sentimen
        # Rule router berfungsi untuk mengarahkan ke rule yang tepat berdasarkan POS tag dan posisi kata dalam kalimat.
        if pos[idx] == "Verba":
            return self.verbRule(idx, words, pos)
        if pos[idx] == "Preposisi":
            return self.prepositionRule(idx, words, pos)
        return idx, self.singleRule(words[idx])

    def removeAffixNya(self, sentence):
        processed_words = []
        for word in sentence.split():
            if "harus" in word:
                processed_words.append(word)
            else:
                processed_words.append(word.replace("nya", ""))
        return " ".join(processed_words)

    def removeMentions(self, sentence):
        filtered_words = [
            word
            for word in sentence.split()
            if "@" not in word and "http" not in word and "https" not in word
        ]
        return self.removeAffixNya(" ".join(filtered_words))

    def dotAndCommaBreak(self, sentence):
        cleaned = (
            self.removeMentions(sentence.lower())
            .replace(".", ",")
            .replace("!", ",")
            .replace("?", ",")
            .replace("#", "")
            .replace("rt", "")
        )
        return [part.strip() for part in cleaned.split(",") if part.strip()]

    def terimakasihPosition(self, sentence):
        # Bagian dari Rule 7:
        # frasa "terima kasih" bernilai positif bila muncul di awal klausa.
        words = sentence.split()
        if not words:
            return 0
        if words[0] == "terimakasih":
            return 1
        if len(words) > 1 and words[0] == "terima" and words[1] == "kasih":
            return 1
        return 0

    def anyFrase(self, sentence):
        # Rule 7:
        # mendeteksi frasa pada kamus frasa sebagai satu kesatuan makna.
        if self.frasadf.empty:
            return []

        preponegatif = {"tidak", "belum", "anti", "bukan"}
        found = []
        tokens = sentence.split()
        used_indexes = set()

        for length in range(self.max_phrase_len, 1, -1):
            if len(tokens) < length:
                continue

            for idx in range(len(tokens) - length + 1):
                phrase_indexes = set(range(idx, idx + length))
                if phrase_indexes.intersection(used_indexes):
                    continue

                phrase = " ".join(tokens[idx : idx + length])
                if phrase in self.frasa_lookup:
                    sentimentfrase = int(self.frasa_lookup[phrase])
                    if idx > 0 and tokens[idx - 1] in preponegatif:
                        sentimentfrase = self.sentimentANDOperator(sentimentfrase, -1)
                    found.append([phrase, sentimentfrase])
                    used_indexes.update(phrase_indexes)
        return found

    def checkFrase(self, sentence):
        # Rule 7:
        # memberi skor frasa lalu menghapusnya dari kalimat agar tidak dihitung dua kali.
        check = self.anyFrase(sentence)
        sentiment = 0
        if check:
            for item in check:
                sentence = sentence.replace(str(item[0]), "")
                sentiment += item[1]
            return sentiment, sentence
        return 0, sentence

    def normalizeSentimentVal(self, val):
        if val == 0:
            return 0
        if val > 0:
            return 1
        return -1

    def label_from_score(self, score):
        if score > 0:
            return "Positif"
        if score < 0:
            return "Negatif"
        return "Netral"

    def _prepare_text(self, sentence):
        if sentence is None:
            return ""
        if isinstance(sentence, list):
            sentence = " ".join(str(token) for token in sentence)
        sentence = str(sentence).strip().lower()
        sentence = re.sub(r"\s+", " ", sentence)
        return sentence

    def _is_question_text(self, sentence):
        # Rule 9:
        # komentar pertanyaan cenderung netral bila tidak ada sinyal negatif kuat.
        if "?" in sentence:
            return True
        tokens = sentence.split()
        return any(token in self.question_words for token in tokens)

    def _has_strong_negative_signal(self, sentence):
        # Rule 8:
        # mendeteksi frasa/kata negatif kuat agar keluhan tegas tidak jatuh ke netral.
        if any(phrase in sentence for phrase in self.strong_negative_phrases):
            return True
        tokens = sentence.split()
        if any(token in self.strong_negative_words for token in tokens):
            return True

        neg_patterns = [
            ("tidak", "ada"),
            ("tidak", "respon"),
            ("tidak", "tanggapan"),
            ("tidak", "tindak"),
            ("belum", "ada"),
            ("belum", "respon"),
            ("belum", "tanggapan"),
            ("belum", "tindak"),
        ]
        token_pairs = set(zip(tokens, tokens[1:])) if len(tokens) > 1 else set()
        if any(pattern in token_pairs for pattern in neg_patterns):
            return True

        issue_words = {
            "air",
            "meter",
            "tagih",
            "tagihan",
            "tarif",
            "bocor",
            "keruh",
            "kotor",
            "mati",
            "keluar",
            "alir",
            "hidup",
            "jalan",
            "naik",
            "lonjak",
            "melonjak",
            "drastis",
        }
        if any(token in self.complaint_words for token in tokens) and (
            any(token in issue_words for token in tokens) or any(token in self.negation_words for token in tokens)
        ):
            return True

        return False

    def _complaint_context_penalty(self, words):
        # Rule 10:
        # memberi penalti saat konteks keluhan bertemu kata-kata issue layanan.
        complaint_present = any(word in self.complaint_words for word in words)
        issue_words = {
            "air",
            "meter",
            "tagih",
            "tagihan",
            "tarif",
            "bocor",
            "keruh",
            "kotor",
            "mati",
            "keluar",
            "alir",
            "hidup",
            "jalan",
            "rusak",
            "macet",
            "kecil",
            "naik",
            "lonjak",
            "melonjak",
            "drastis",
            "ubah",
        }
        issue_present = any(word in issue_words for word in words)
        negation_present = any(word in self.negation_words for word in words)
        delay_patterns = {
            ("sampai", "sekarang"),
            ("belum", "ada"),
            ("tidak", "ada"),
            ("tidak", "keluar"),
            ("tidak", "jalan"),
        }
        token_pairs = set(zip(words, words[1:])) if len(words) > 1 else set()
        billing_words = {"tagih", "tagihan", "tarif", "bayar", "biaya", "ribu", "kode"}
        anomaly_words = {"naik", "lonjak", "melonjak", "drastis", "ubah", "berubah"}

        if complaint_present and issue_present:
            return -1
        if negation_present and issue_present:
            return -1
        if any(pattern in token_pairs for pattern in delay_patterns):
            return -1
        if any(word in billing_words for word in words) and any(word in anomaly_words for word in words):
            return -1
        return 0

    def getSentimentScore(self, sentence):
        # Orkestrasi rule:
        # Rule 9 diperiksa lebih awal, lalu Rule 7/8/10 dan rule POS 1-6 dijumlahkan.
        totalsentiment = 0
        sentence = self._prepare_text(sentence)
        if not sentence:
            return 0

        if self._is_question_text(sentence) and not self._has_strong_negative_signal(sentence):
            return 0

        sentencebreak = self.dotAndCommaBreak(sentence)

        for istc, sentenceb in enumerate(sentencebreak):
            skipIndex = []
            sentimentval = 0

            if self.terimakasihPosition(sentenceb) == 1 and istc == 0:
                # Rule 7: bonus positif untuk "terima kasih" di awal klausa.
                sentenceb = sentenceb.replace("terima", "").replace("kasih", "").strip()
                sentimentval += 1

            sentimentfrase, sentenceb = self.checkFrase(sentenceb)
            sentimentval += sentimentfrase

            words = [item for item in sentenceb.split() if item and item not in self.sanitisinglist]
            sentimentval += self._complaint_context_penalty(words)
            pos = self.convertSentence(" ".join(words))

            for idx, tag in enumerate(pos):
                if idx in skipIndex or tag not in self.pattern:
                    continue

                if words[idx] in self.negation_words and idx + 1 < len(words):
                    # Shortcut implementasi untuk pola negasi yang praktis mewakili
                    # Rule 4 dan pada beberapa kasus membantu Rule 6.
                    next_sentiment = self.singleRule(words[idx + 1])
                    if next_sentiment != 0:
                        sentimentval += self.sentimentANDOperator(-1, next_sentiment)
                        skipIndex.extend([idx, idx + 1])
                        continue

                issentiment, sentiment = self.getWordSentimentValue(idx, words, pos)
                sentimentval += sentiment

                if isinstance(issentiment, list):
                    skipIndex.extend(issentiment)
                elif issentiment is not None:
                    skipIndex.append(issentiment)

            totalsentiment += sentimentval

        return int(totalsentiment)

    def getSentiment(self, sentence):
        return self.normalizeSentimentVal(self.getSentimentScore(sentence))

    def predict(self, sentence):
        score = self.getSentimentScore(sentence)
        normalized = self.normalizeSentimentVal(score)
        return {
            "text": sentence,
            "score": score,
            "normalized_score": normalized,
            "label": self.label_from_score(normalized),
        }

    def predict_series(self, texts):
        return pd.DataFrame([self.predict(text) for text in texts])

    def apply_to_dataframe(
        self,
        df,
        text_column="stemmed_normalized",
        score_column="rule_based_sentiment_score",
        normalized_column="rule_based_sentiment_normalized",
        label_column="rule_based_sentiment_label",
    ):
        result = df.copy()
        result[score_column] = result[text_column].apply(self.getSentimentScore)
        result[normalized_column] = result[score_column].apply(self.normalizeSentimentVal)
        result[label_column] = result[normalized_column].apply(self.label_from_score)
        return result
