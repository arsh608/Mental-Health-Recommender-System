"""
tfidf_engine.py
TF-IDF + Cosine Similarity content retrieval engine.
Indexes the content library on first load and caches the matrix.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "models" / "saved"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class ContentTFIDF:
    CACHE_PATH = CACHE_DIR / "content_tfidf.pkl"

    def __init__(self):
        self.df: pd.DataFrame = None
        self.vectorizer: TfidfVectorizer = None
        self.matrix = None
        self._loaded = False

    def _combined_text(self, row: pd.Series) -> str:
        parts = [
            str(row.get("title", "")),
            str(row.get("tags", "")),
            str(row.get("description", "")),
            str(row.get("mood_tags", "")),
            # repeat title and tags to boost their weight
            str(row.get("title", "")),
            str(row.get("tags", "")),
        ]
        return " ".join(p for p in parts if p and p != "nan")

    def load(self):
        if self._loaded:
            return

        csv_path = DATA_DIR / "content_library.csv"
        self.df = pd.read_csv(csv_path)
        self.df["combined_text"] = self.df.apply(self._combined_text, axis=1)

        if self.CACHE_PATH.exists():
            try:
                ckpt = joblib.load(self.CACHE_PATH)
                self.vectorizer = ckpt["vectorizer"]
                self.matrix = ckpt["matrix"]
                self._loaded = True
                return
            except Exception:
                pass

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,
            stop_words="english",
        )
        self.matrix = self.vectorizer.fit_transform(self.df["combined_text"])
        joblib.dump({"vectorizer": self.vectorizer, "matrix": self.matrix},
                    self.CACHE_PATH)
        self._loaded = True

    def search(self, query: str, top_n: int = 30) -> pd.DataFrame:
        self.load()
        print(f"[TFIDF] Searching for: {query[:50]}...")
        print(f"[TFIDF] Vectorizer: {self.vectorizer}, Matrix shape: {self.matrix.shape if self.matrix is not None else 'None'}")
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        idx = sims.argsort()[::-1][:top_n]
        result = self.df.iloc[idx].copy()
        result["cosine_score"] = sims[idx]
        print(f"[TFIDF] Returning {len(result)} results")
        return result

    def get_all(self) -> pd.DataFrame:
        self.load()
        return self.df.copy()


_engine = ContentTFIDF()

def get_engine() -> ContentTFIDF:
    return _engine
