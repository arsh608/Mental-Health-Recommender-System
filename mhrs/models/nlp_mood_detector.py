"""
nlp_mood_detector.py
─────────────────────────────────────────────────────────────────────────────
Multi-layer NLP mood detection pipeline:

  Layer 1 – VADER Sentiment Analysis
             Fast lexicon-based sentiment (positive/negative/neutral scores).

  Layer 2 – TextBlob Subjectivity & Polarity
             Secondary sentiment confirmation.

  Layer 3 – TF-IDF + Logistic Regression Classifier
             Trained on the unified Kaggle dataset (emotions + counseling).
             Handles the 9 fine-grained mood classes.

  Layer 4 – Sentence-Transformer Zero-Shot Semantic Similarity
             Maps user text to mood prototype embeddings.
             Most accurate but slowest; used as top-up signal.

  Fusion   – Weighted ensemble of all four layers.
             Weights: VADER=0.10, TextBlob=0.05, TF-IDF=0.40, SentBERT=0.45
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import os, sys, re, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
from functools import lru_cache

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))

ALL_MOODS = [
    "anxiety", "stress", "sadness", "anger",
    "low_motivation", "loneliness", "insomnia",
    "overwhelmed", "low_self_esteem", "positive",
]

MOOD_DISPLAY = {
    "anxiety":        "Anxious / Worried",
    "stress":         "Stressed / Burned Out",
    "sadness":        "Sad / Depressed",
    "anger":          "Frustrated / Angry",
    "low_motivation": "Low Motivation",
    "loneliness":     "Lonely / Isolated",
    "insomnia":       "Sleep Problems",
    "overwhelmed":    "Overwhelmed",
    "low_self_esteem":"Low Self-Esteem",
    "positive":       "Doing Well",
}

MOOD_EMOJI = {
    "anxiety": "😰", "stress": "😤", "sadness": "😢",
    "anger": "😠", "low_motivation": "😴", "loneliness": "🥺",
    "insomnia": "🌙", "overwhelmed": "🌊", "low_self_esteem": "💙",
    "positive": "😊",
}

MOOD_COLOR = {
    "anxiety": "#7F77DD", "stress": "#E8593C", "sadness": "#378ADD",
    "anger": "#D85A30", "low_motivation": "#639922", "loneliness": "#1D9E75",
    "insomnia": "#534AB7", "overwhelmed": "#D4537E",
    "low_self_esteem": "#185FA5", "positive": "#3B6D11",
}

# ── Mood prototype sentences for zero-shot semantic similarity ────────────────
MOOD_PROTOTYPES: Dict[str, List[str]] = {
    "anxiety": [
        "I feel anxious and constantly worried about things that might go wrong",
        "My mind races with anxious thoughts and I cannot stop overthinking",
        "I have panic attacks and feel intense fear and dread regularly",
        "Anxiety is taking over my life and I feel nervous all the time",
        "I am terrified of the future and dread what might happen next",
        "My heart races and I feel scared even when there is no danger",
        "I worry obsessively and catastrophise about everything",
    ],
    "stress": [
        "I feel stressed and overwhelmed by too many responsibilities",
        "Work pressure is crushing me and I am completely burned out",
        "I have too many deadlines and cannot cope with the pressure",
        "I feel frantic and hectic, running from task to task with no rest",
        "The stress is affecting my health and I feel physically tense",
        "I am at my limit and feel like I cannot take any more stress",
        "Everything is piling up and I cannot see a way through",
    ],
    "sadness": [
        "I feel deeply sad and hopeless, like things will never improve",
        "I am depressed and feel empty, numb, and disconnected from life",
        "I keep crying and feel a heavy grief that will not lift",
        "Everything feels pointless and I have lost the will to try",
        "I feel heartbroken and miserable and cannot find any joy",
        "The depression makes me feel worthless and without purpose",
        "I feel a deep aching sadness that permeates everything I do",
    ],
    "anger": [
        "I am very angry and frustrated, everything feels deeply unfair",
        "I feel intense rage and resentment that I cannot control",
        "I keep snapping at people and feel irritable all the time",
        "The anger is consuming me and I want to explode",
        "I feel furious about the injustice of my situation",
        "My frustration has built into rage and I feel out of control",
        "I am hostile and bitter towards people and situations around me",
    ],
    "low_motivation": [
        "I have no motivation at all and cannot get myself to do anything",
        "I keep procrastinating on important tasks and feel stuck",
        "I have lost all drive and nothing excites or interests me",
        "I feel apathetic and cannot bring myself to start anything",
        "Everything feels pointless and I have no energy to act",
        "I am stuck in a rut and cannot find any reason to move forward",
        "My motivation has completely disappeared and I feel flat",
    ],
    "loneliness": [
        "I feel very lonely and isolated, like no one understands me",
        "I have no real connections and feel completely alone in the world",
        "I feel invisible and like nobody cares about my existence",
        "I am disconnected from everyone and feel deeply alone",
        "I have no friends and the loneliness is painful and constant",
        "I feel rejected and abandoned and cannot connect with anyone",
        "The isolation is crushing and I feel unseen and unheard",
    ],
    "insomnia": [
        "I cannot sleep and lie awake for hours with my mind racing",
        "My sleep is terrible and I wake up exhausted every morning",
        "I keep waking up through the night and cannot get rest",
        "I dread bedtime because I know I will not be able to sleep",
        "Insomnia is ruining my mental and physical health",
        "I feel exhausted but my mind will not let me rest at night",
        "I have not slept properly in weeks and it is affecting everything",
    ],
    "overwhelmed": [
        "I feel completely overwhelmed and like everything is too much",
        "I am at my breaking point and cannot handle any more",
        "Everything is falling apart and I feel like I am sinking",
        "I have lost control of everything and feel desperate",
        "The weight of everything is crushing me and I cannot breathe",
        "I am drowning in responsibilities and do not know what to do",
        "I feel paralysed by the enormity of what I need to handle",
    ],
    "low_self_esteem": [
        "I feel worthless and like I am not good enough for anything",
        "I hate myself and constantly compare myself to others",
        "I am ashamed of who I am and feel deeply inadequate",
        "I have no confidence and feel like a complete failure",
        "I believe I am fundamentally flawed and unlovable",
        "My self-esteem is crushed and I feel inferior to everyone",
        "I cannot accept compliments because I do not believe I deserve them",
    ],
    "positive": [
        "I feel good and content with how things are going in my life",
        "I am doing well and feeling calm, grateful, and at peace",
        "I feel motivated and optimistic about what the future holds",
        "Things are getting better and I feel hopeful and energised",
        "I am grateful for my life and feel genuinely happy today",
        "I feel resilient and capable and ready to face challenges",
        "Life feels manageable and I am in a good place emotionally",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: VADER
# ─────────────────────────────────────────────────────────────────────────────

class VADERLayer:
    def __init__(self):
        self._analyzer = None

    def load(self):
        try:
            import nltk
            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self._analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            print(f"[VADER] Load failed: {e}")

    def score(self, text: str) -> Dict[str, float]:
        """
        Returns mood scores inferred from VADER sentiment.
        Negative sentiment → raises scores for anxiety/sadness/anger/stress.
        """
        if self._analyzer is None:
            return {m: 0.0 for m in ALL_MOODS}

        vs = self._analyzer.polarity_scores(text)
        neg = vs["neg"]
        pos = vs["pos"]
        compound = vs["compound"]  # -1 to +1

        scores = {m: 0.0 for m in ALL_MOODS}

        if compound >= 0.3:
            scores["positive"] = min(pos * 1.5, 1.0)
        elif compound <= -0.5:
            # Very negative: distribute across negative moods
            scores["sadness"]    = neg * 0.35
            scores["anxiety"]    = neg * 0.25
            scores["stress"]     = neg * 0.20
            scores["anger"]      = neg * 0.10
            scores["overwhelmed"]= neg * 0.10
        elif compound < 0:
            scores["sadness"]    = neg * 0.25
            scores["stress"]     = neg * 0.20
            scores["anxiety"]    = neg * 0.20
            scores["anger"]      = neg * 0.15
            scores["overwhelmed"]= neg * 0.20

        return scores


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: TextBlob
# ─────────────────────────────────────────────────────────────────────────────

class TextBlobLayer:
    def score(self, text: str) -> Dict[str, float]:
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            polarity    = blob.sentiment.polarity     # -1 to +1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
        except Exception:
            return {m: 0.0 for m in ALL_MOODS}

        scores = {m: 0.0 for m in ALL_MOODS}
        if polarity > 0.2:
            scores["positive"] = polarity * 0.8
        elif polarity < -0.1:
            scores["sadness"] = abs(polarity) * 0.4
            scores["stress"]  = abs(polarity) * 0.3
            scores["anxiety"] = abs(polarity) * 0.3
        return scores


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: TF-IDF + Logistic Regression (trained on Kaggle data)
# ─────────────────────────────────────────────────────────────────────────────

class TFIDFClassifierLayer:
    MODEL_PATH = MODEL_DIR / "tfidf_logreg.pkl"

    def __init__(self):
        self.pipeline = None
        self.classes_: List[str] = []

    def _build_pipeline(self):
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        return Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=30000,
                sublinear_tf=True,
                min_df=2,
                analyzer="word",
                token_pattern=r"\b[a-z][a-z]+\b",
            )),
            ("clf", LogisticRegression(
                C=5.0,
                max_iter=1000,
                multi_class="multinomial",
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
            )),
        ])

    def train(self, df: pd.DataFrame):
        """Train on the unified Kaggle dataset."""
        from sklearn.preprocessing import LabelEncoder

        print("[TFIDFLayer] Training TF-IDF + LogReg classifier…")
        X = df["text"].tolist()
        y = df["mood"].tolist()

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X, y)
        self.classes_ = list(self.pipeline.classes_)

        joblib.dump({"pipeline": self.pipeline, "classes": self.classes_},
                    self.MODEL_PATH)
        print(f"[TFIDFLayer] Model saved → {self.MODEL_PATH}")

        # Quick accuracy on training data
        preds = self.pipeline.predict(X[:500])
        acc = np.mean(np.array(preds) == np.array(y[:500]))
        print(f"[TFIDFLayer] Train accuracy (sample): {acc:.3f}")

    def load(self):
        if self.MODEL_PATH.exists():
            ckpt = joblib.load(self.MODEL_PATH)
            self.pipeline = ckpt["pipeline"]
            self.classes_ = ckpt["classes"]
            print("[TFIDFLayer] Model loaded from cache.")
            return True
        return False

    def score(self, text: str) -> Dict[str, float]:
        if self.pipeline is None:
            return {m: 0.0 for m in ALL_MOODS}
        proba = self.pipeline.predict_proba([text])[0]
        raw = dict(zip(self.classes_, proba))
        # Ensure all moods present
        return {m: float(raw.get(m, 0.0)) for m in ALL_MOODS}


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4: Sentence-Transformer Zero-Shot
# ─────────────────────────────────────────────────────────────────────────────

class SentenceBERTLayer:
    PROTO_CACHE = MODEL_DIR / "proto_embeddings.npy"

    def __init__(self):
        self.model = None
        self.proto_embs: Dict[str, np.ndarray] = {}

    def load(self):
        if self.model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            print("[SentBERT] Loading all-MiniLM-L6-v2…")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self._compute_prototypes()
            print("[SentBERT] Ready.")
        except Exception as e:
            print(f"[SentBERT] Load failed: {e}")

    def _compute_prototypes(self):
        if self.PROTO_CACHE.exists():
            data = np.load(self.PROTO_CACHE, allow_pickle=True).item()
            self.proto_embs = data
            return
        print("[SentBERT] Pre-computing prototype embeddings…")
        for mood, sentences in MOOD_PROTOTYPES.items():
            embs = self.model.encode(sentences, convert_to_numpy=True,
                                     show_progress_bar=False)
            self.proto_embs[mood] = embs
        np.save(self.PROTO_CACHE, self.proto_embs)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def score(self, text: str) -> Dict[str, float]:
        if self.model is None or not self.proto_embs:
            return {m: 0.0 for m in ALL_MOODS}
        try:
            # Multi-threading/vectorization happens inside encode
            emb = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
            
            raw = {}
            # Pre-calculate norms for all prototype embeddings once or use dot product if normalized
            # For simplicity, we'll use a slightly more vectorized approach than before
            for mood, proto_embs in self.proto_embs.items():
                # proto_embs is (N, D), emb is (D,)
                # compute cosine similarity for all prototypes of this mood at once
                dot = np.dot(proto_embs, emb)
                norms = np.linalg.norm(proto_embs, axis=1) * np.linalg.norm(emb) + 1e-9
                sims = dot / norms
                
                # Take top-3 average
                raw[mood] = float(np.mean(np.sort(sims)[-3:]))

            # Soft-max normalise to boost the signal
            vals = np.array([raw[m] for m in ALL_MOODS])
            # Apply a small temperature to sharpen the distribution
            vals = np.exp((vals - np.mean(vals)) / (np.std(vals) + 1e-9))
            vals = vals / (np.sum(vals) + 1e-9)
            
            return {m: float(v) for m, v in zip(ALL_MOODS, vals)}
        except Exception as e:
            print(f"[SentBERT] Inference error: {e}")
            return {m: 0.0 for m in ALL_MOODS}


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Detector
# ─────────────────────────────────────────────────────────────────────────────

class NLPMoodDetector:
    """
    Full ensemble mood detector. Call load() once, then detect() for each query.
    """

    WEIGHTS = {
        "vader":    0.10,
        "textblob": 0.05,
        "tfidf":    0.40,
        "sentbert": 0.45,
    }

    def __init__(self):
        self.vader     = VADERLayer()
        self.textblob  = TextBlobLayer()
        self.tfidf     = TFIDFClassifierLayer()
        self.sentbert  = SentenceBERTLayer()
        self._ready    = False

    # ── Setup ────────────────────────────────────────────────────────────────

    def load(self, force_retrain: bool = False):
        """Load or train all layers."""
        print("[NLPMoodDetector] Initialising…")

        # Layer 1 & 2: always fast
        self.vader.load()

        # Layer 3: TF-IDF — train if no saved model
        if force_retrain or not self.tfidf.load():
            from data.data_loader import build_unified_training_data
            df = build_unified_training_data()
            self.tfidf.train(df)

        # Layer 4: Sentence-BERT
        self.sentbert.load()

        self._ready = True
        print("[NLPMoodDetector] All layers ready.")

    # ── Preprocessing ────────────────────────────────────────────────────────

    @staticmethod
    def preprocess(text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^\w\s'\-]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    # ── Core detection ────────────────────────────────────────────────────────

    @lru_cache(maxsize=200)
    def detect(
        self,
        text: str,
    ) -> Dict:
        """
        Detect mood from free-form text.

        Returns dict with:
          primary_mood  : str
          confidence    : float 0–1
          all_scores    : dict mood → ensemble score
          layer_scores  : dict layer → mood scores
          sentiment     : dict with polarity/subjectivity
          top_moods     : list of (mood, score) sorted desc
        """
        if not self._ready:
            self.load()

        clean = self.preprocess(text)

        # Run all layers
        v_scores  = self.vader.score(clean)
        tb_scores = self.textblob.score(clean)
        tf_scores = self.tfidf.score(clean)
        sb_scores = self.sentbert.score(clean)

        # Weighted ensemble
        ensemble = {}
        for mood in ALL_MOODS:
            ensemble[mood] = (
                self.WEIGHTS["vader"]    * v_scores[mood]  +
                self.WEIGHTS["textblob"] * tb_scores[mood] +
                self.WEIGHTS["tfidf"]    * tf_scores[mood] +
                self.WEIGHTS["sentbert"] * sb_scores[mood]
            )

        # Normalise to sum to 1
        total = sum(ensemble.values()) + 1e-9
        ensemble = {k: v / total for k, v in ensemble.items()}

        primary = max(ensemble, key=ensemble.get)
        confidence = ensemble[primary]
        top_moods = sorted(ensemble.items(), key=lambda x: x[1], reverse=True)

        # Sentiment summary from TextBlob
        sentiment = {"polarity": 0.0, "subjectivity": 0.0}
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            sentiment = {
                "polarity":     round(blob.sentiment.polarity, 3),
                "subjectivity": round(blob.sentiment.subjectivity, 3),
            }
        except Exception:
            pass

        return {
            "primary_mood": primary,
            "confidence":   round(confidence, 4),
            "all_scores":   {k: round(v, 4) for k, v in ensemble.items()},
            "layer_scores": {
                "vader":    {k: round(v, 4) for k, v in v_scores.items()},
                "textblob": {k: round(v, 4) for k, v in tb_scores.items()},
                "tfidf":    {k: round(v, 4) for k, v in tf_scores.items()},
                "sentbert": {k: round(v, 4) for k, v in sb_scores.items()},
            },
            "sentiment": sentiment,
            "top_moods": [(m, round(s, 4)) for m, s in top_moods],
            "display_name": MOOD_DISPLAY.get(primary, primary),
            "emoji":        MOOD_EMOJI.get(primary, ""),
            "color":        MOOD_COLOR.get(primary, "#6b7280"),
        }

    def batch_detect(self, texts: List[str]) -> List[Dict]:
        return [self.detect(t) for t in texts]


# ── Singleton ─────────────────────────────────────────────────────────────────
_detector: Optional[NLPMoodDetector] = None

def get_detector() -> NLPMoodDetector:
    global _detector
    if _detector is None:
        _detector = NLPMoodDetector()
    return _detector


if __name__ == "__main__":
    detector = NLPMoodDetector()
    detector.load()

    test_inputs = [
        "I have been feeling really anxious lately, my mind races and I can't stop worrying",
        "I am completely burned out from work, too many deadlines and not enough time",
        "I feel so sad and hopeless, nothing seems to get better and I cry most days",
        "I am so angry and frustrated, nothing is fair and I want to explode",
        "I have zero motivation and keep procrastinating on everything",
        "I feel completely alone and like no one understands what I am going through",
        "I can't sleep at all, I lie awake for hours with racing thoughts",
        "Everything is too much, I am overwhelmed and at my breaking point",
        "I hate myself and feel like such a failure compared to everyone else",
        "I feel good today, grateful and content with life",
    ]

    print("\n" + "="*70)
    print("NLP MOOD DETECTOR — TEST RESULTS")
    print("="*70)
    for text in test_inputs:
        result = detector.detect(text)
        print(f"\nInput: {text[:65]}…")
        print(f"  → {result['emoji']} {result['display_name']} "
              f"(confidence: {result['confidence']*100:.1f}%)")
        print(f"  → Top 3: {[(MOOD_DISPLAY[m], f'{s*100:.1f}%') for m,s in result['top_moods'][:3]]}")
        print(f"  → Sentiment: polarity={result['sentiment']['polarity']}, "
              f"subjectivity={result['sentiment']['subjectivity']}")
