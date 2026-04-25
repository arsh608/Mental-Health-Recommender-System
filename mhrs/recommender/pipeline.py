"""
pipeline.py
Main entry point: mood detection → retrieval → ranking.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from typing import Dict, Optional, Tuple

from models.nlp_mood_detector import get_detector, MOOD_DISPLAY, MOOD_EMOJI, MOOD_COLOR
from recommender.tfidf_engine import get_engine
from recommender.mood_mapper import get_mood_config, rank_candidates


def load_all():
    """Pre-warm all models. Call once at app startup."""
    print("[Pipeline] Loading NLP detector…")
    get_detector().load()
    print("[Pipeline] Loading content index…")
    get_engine().load()
    print("[Pipeline] Ready.")


def recommend(
    user_text: str,
    stress_level: int = 5,
    content_type_filter: Optional[str] = None,
    top_n: int = 8,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Full pipeline.

    Returns:
        mood_result : dict from NLPMoodDetector.detect()
        recommendations : ranked DataFrame
    """
    detector = get_detector()
    if not detector._ready:
        detector.load()

    engine = get_engine()
    engine.load()

    # 1. Detect mood
    mood_result = detector.detect(user_text)
    primary_mood = mood_result["primary_mood"]

    # 2. Build enriched query
    cfg = get_mood_config(primary_mood)
    query = cfg["query"] + " " + user_text

    # 3. Retrieve candidates
    candidates = engine.search(query, top_n=min(top_n * 4, 60))

    # 4. Filter by type
    if content_type_filter and content_type_filter.lower() not in ("all", ""):
        filtered = candidates[candidates["type"].str.lower() == content_type_filter.lower()]
        candidates = filtered if len(filtered) >= 3 else candidates

    # 5. Rank
    ranked = rank_candidates(candidates, primary_mood, stress_level=stress_level)

    return mood_result, ranked.head(top_n)


def get_all_content() -> pd.DataFrame:
    engine = get_engine()
    engine.load()
    return engine.get_all()
