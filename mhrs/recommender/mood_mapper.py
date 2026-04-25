"""
mood_mapper.py  +  ranker.py  (combined)
─────────────────────────────────────────────────────────────────────────────
Mood → content tag mapping and weighted ranking algorithm.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

# ── Mood configuration ────────────────────────────────────────────────────────

MOOD_CONFIG: Dict[str, Dict] = {
    "anxiety": {
        "query": "anxiety panic breathing grounding mindfulness calm nervous worry fear",
        "preferred_types": ["exercise", "video", "article"],
        "boost_tags": ["anxiety","grounding","panic","breathing","mindfulness","calm","fear"],
        "avoid_tags": [],
        "description": "Tools to quiet an anxious, racing mind",
        "color": "#7F77DD",
    },
    "stress": {
        "query": "stress relief breathing relaxation burnout decompress rest calm",
        "preferred_types": ["exercise", "video", "article"],
        "boost_tags": ["stress","breathing","calm","relaxation","burnout","rest","decompress"],
        "avoid_tags": [],
        "description": "Resources to decompress and rebuild",
        "color": "#E8593C",
    },
    "sadness": {
        "query": "depression sadness grief hopeless motivation compassion gratitude meaning",
        "preferred_types": ["article", "video", "exercise"],
        "boost_tags": ["sadness","depression","grief","compassion","gratitude","meaning","hope"],
        "avoid_tags": [],
        "description": "Gentle support for low mood and grief",
        "color": "#378ADD",
    },
    "anger": {
        "query": "anger frustration emotion regulation movement exercise catharsis body",
        "preferred_types": ["exercise", "article", "video"],
        "boost_tags": ["anger","frustration","emotions","regulation","body","movement","catharsis"],
        "avoid_tags": [],
        "description": "Ways to process and release anger",
        "color": "#D85A30",
    },
    "low_motivation": {
        "query": "motivation energy activation procrastination dopamine purpose values",
        "preferred_types": ["video", "article", "exercise"],
        "boost_tags": ["motivation","energy","activation","procrastination","dopamine","values","purpose"],
        "avoid_tags": [],
        "description": "A spark to get you moving again",
        "color": "#639922",
    },
    "loneliness": {
        "query": "loneliness connection compassion relationships belonging isolation social",
        "preferred_types": ["article", "video", "exercise"],
        "boost_tags": ["loneliness","connection","compassion","relationships","belonging","social"],
        "avoid_tags": [],
        "description": "Resources around connection and belonging",
        "color": "#1D9E75",
    },
    "insomnia": {
        "query": "sleep insomnia relaxation bedtime routine body-scan CBT-I circadian wind-down",
        "preferred_types": ["exercise", "article", "video"],
        "boost_tags": ["sleep","insomnia","relaxation","bedtime","circadian","CBT","wind-down"],
        "avoid_tags": [],
        "description": "Evidence-based tools to restore sleep",
        "color": "#534AB7",
    },
    "overwhelmed": {
        "query": "overwhelmed quick reset breathing grounding crisis coping one-step",
        "preferred_types": ["exercise", "video", "article"],
        "boost_tags": ["overwhelmed","breathing","grounding","quick","reset","crisis","coping"],
        "avoid_tags": ["advanced"],
        "description": "Fast relief when everything is too much",
        "color": "#D4537E",
    },
    "low_self_esteem": {
        "query": "self-esteem self-compassion confidence CBT shame vulnerability distortions",
        "preferred_types": ["article", "exercise", "video"],
        "boost_tags": ["self-esteem","confidence","compassion","CBT","shame","vulnerability"],
        "avoid_tags": [],
        "description": "Building a kinder relationship with yourself",
        "color": "#185FA5",
    },
    "positive": {
        "query": "wellbeing growth mindfulness gratitude skills resilience flourishing",
        "preferred_types": ["article", "video", "exercise"],
        "boost_tags": ["wellbeing","gratitude","mindfulness","growth","resilience","flourishing"],
        "avoid_tags": [],
        "description": "Keep building on what's working",
        "color": "#3B6D11",
    },
}


def get_mood_config(mood: str) -> Dict:
    return MOOD_CONFIG.get(mood, MOOD_CONFIG["stress"])


# ── Ranker ────────────────────────────────────────────────────────────────────

def _tag_match(row: pd.Series, boost_tags: List[str]) -> float:
    combined = (str(row.get("tags","")) + " " + str(row.get("mood_tags",""))).lower()
    hits = sum(1 for t in boost_tags if t.lower() in combined)
    return min(hits / max(len(boost_tags), 1), 1.0)


def _difficulty_score(row: pd.Series, stress: int) -> float:
    d = str(row.get("difficulty","beginner")).lower()
    if stress >= 7:
        return {"beginner": 1.0, "intermediate": 0.65, "advanced": 0.30}.get(d, 0.8)
    elif stress <= 3:
        return 0.9
    return {"beginner": 0.85, "intermediate": 1.0, "advanced": 0.70}.get(d, 0.85)


def _type_pref_score(row: pd.Series, preferred: List[str]) -> float:
    t = str(row.get("type","")).lower()
    if t in preferred:
        return 1.0 - preferred.index(t) * 0.15
    return 0.45


def rank_candidates(
    candidates: pd.DataFrame,
    mood: str,
    stress_level: int = 5,
    cosine_w: float = 0.45,
    tag_w: float = 0.35,
    type_w: float = 0.12,
    diff_w: float = 0.08,
) -> pd.DataFrame:
    print(f"[RANKER] Ranking {len(candidates)} candidates for mood: {mood}")
    cfg = get_mood_config(mood)
    boost_tags = cfg["boost_tags"]
    preferred  = cfg["preferred_types"]

    df = candidates.copy()
    cos_max = df["cosine_score"].max() if df["cosine_score"].max() > 0 else 1
    df["cosine_norm"]     = df["cosine_score"] / cos_max
    df["tag_score"]       = df.apply(lambda r: _tag_match(r, boost_tags), axis=1)
    df["type_score"]      = df.apply(lambda r: _type_pref_score(r, preferred), axis=1)
    df["difficulty_score"]= df.apply(lambda r: _difficulty_score(r, stress_level), axis=1)

    df["final_score"] = (
        cosine_w * df["cosine_norm"] +
        tag_w    * df["tag_score"]   +
        type_w   * df["type_score"]  +
        diff_w   * df["difficulty_score"]
    )
    result = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    print(f"[RANKER] Returning {len(result)} ranked results")
    return result
