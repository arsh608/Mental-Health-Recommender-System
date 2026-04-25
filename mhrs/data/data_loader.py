"""
data_loader.py
─────────────────────────────────────────────────────────────────────────────
Downloads, preprocesses, and caches both Kaggle datasets:

  1. praveengovi/emotions-dataset-for-nlp
     → train.txt / val.txt / test.txt  (text;label format)
     → 6 classes: joy, sadness, anger, fear, love, surprise

  2. melissamonfared/mental-health-counseling-conversations-k
     → Counseling_Data.csv  (questionText, answerText)

Also builds emotion → mood mapping and exports unified processed CSV.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import json
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

for d in [DATA_DIR, RAW_DIR, PROC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Emotion → unified mood mapping ───────────────────────────────────────────
# The Kaggle dataset has 6 labels; we expand to our 9 moods via rules below.
EMOTION_TO_MOOD: Dict[str, str] = {
    "joy":      "positive",
    "love":     "positive",
    "surprise": "positive",
    "sadness":  "sadness",
    "anger":    "anger",
    "fear":     "anxiety",
}

# Extended mood labels we detect (beyond the 6 Kaggle classes)
ALL_MOODS = [
    "anxiety", "stress", "sadness", "anger",
    "low_motivation", "loneliness", "insomnia",
    "overwhelmed", "low_self_esteem", "positive",
]

# ── Keyword-based mood augmentation for unlabelled counseling data ────────────
MOOD_KEYWORD_SETS: Dict[str, List[str]] = {
    "anxiety": [
        "anxious","anxiety","panic","worry","worried","nervous","fear","scared",
        "racing thoughts","overthink","dread","terrified","phobia","afraid",
        "uneasy","apprehensive","restless","on edge","tense","heart racing",
    ],
    "stress": [
        "stress","stressed","overwhelmed","pressure","deadline","burnout",
        "burned out","exhausted","too much","hectic","frantic","overloaded",
        "swamped","can't cope","losing it","no time","juggling","breaking point",
    ],
    "sadness": [
        "sad","sadness","depressed","depression","hopeless","empty","numb",
        "crying","grief","loss","lonely","miserable","heartbroken","worthless",
        "meaningless","gloomy","blue","down","no point","nothing matters",
    ],
    "anger": [
        "angry","anger","furious","mad","rage","frustrated","frustration",
        "irritated","annoyed","resentful","bitter","hate","unfair","livid",
        "fuming","outraged","hostile","aggressive","snap","yelling",
    ],
    "low_motivation": [
        "unmotivated","procrastinating","procrastination","no motivation",
        "stuck","paralysed","apathetic","apathy","pointless","giving up",
        "no energy","lethargic","sluggish","can't start","don't care","bored",
    ],
    "loneliness": [
        "lonely","loneliness","alone","isolated","no friends","disconnected",
        "invisible","no one cares","no one understands","left out","excluded",
        "abandoned","rejected","unwanted","forgotten","by myself","isolated",
    ],
    "insomnia": [
        "can't sleep","insomnia","sleepless","tossing","tired","awake",
        "waking up","nightmares","bad dreams","not rested","fatigue",
        "3am","night","exhausted but can't",
    ],
    "overwhelmed": [
        "overwhelmed","too much","can't handle","falling apart","losing it",
        "spinning","drowning","sinking","buried","suffocating","out of control",
        "chaos","everything at once","no control","breaking point",
    ],
    "low_self_esteem": [
        "worthless","useless","failure","stupid","loser","hate myself",
        "not good enough","inadequate","insecure","ashamed","humiliated",
        "comparing","imposter","fraud","not worthy","self-hatred","ugly",
    ],
    "positive": [
        "happy","joyful","excited","motivated","grateful","thankful","better",
        "hopeful","content","peaceful","calm","good","great","wonderful",
        "blessed","okay","fine","improving","healing","getting better",
    ],
}


def label_mood_from_keywords(text: str) -> str:
    """Assign a mood label to free text using keyword voting."""
    text_lower = text.lower()
    scores = {mood: 0 for mood in ALL_MOODS}
    for mood, keywords in MOOD_KEYWORD_SETS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[mood] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "stress"  # default


# ── Download helpers ──────────────────────────────────────────────────────────

def _kaggle_available() -> bool:
    try:
        import kaggle
        return True
    except Exception:
        return False


def download_emotions_dataset() -> Path:
    """
    Download praveengovi/emotions-dataset-for-nlp from Kaggle.
    Falls back to a bundled synthetic sample if Kaggle credentials are absent.
    """
    out_dir = RAW_DIR / "emotions"
    flag = out_dir / ".downloaded"
    if flag.exists():
        print("[DataLoader] Emotions dataset already downloaded.")
        return out_dir

    out_dir.mkdir(exist_ok=True)

    if _kaggle_available():
        try:
            import kaggle
            print("[DataLoader] Downloading emotions dataset from Kaggle…")
            kaggle.api.dataset_download_files(
                "praveengovi/emotions-dataset-for-nlp",
                path=str(out_dir),
                unzip=True,
            )
            flag.touch()
            print("[DataLoader] Emotions dataset downloaded.")
            return out_dir
        except Exception as e:
            print(f"[DataLoader] Kaggle download failed: {e}")

    # ── Fallback: create synthetic representative dataset ──────────────────
    print("[DataLoader] Using built-in representative sample…")
    _write_synthetic_emotions(out_dir)
    flag.touch()
    return out_dir


def download_counseling_dataset() -> Path:
    """
    Download melissamonfared/mental-health-counseling-conversations-k.
    Falls back to built-in sample if credentials absent.
    """
    out_dir = RAW_DIR / "counseling"
    flag = out_dir / ".downloaded"
    if flag.exists():
        print("[DataLoader] Counseling dataset already downloaded.")
        return out_dir

    out_dir.mkdir(exist_ok=True)

    if _kaggle_available():
        try:
            import kaggle
            print("[DataLoader] Downloading counseling dataset from Kaggle…")
            kaggle.api.dataset_download_files(
                "melissamonfared/mental-health-counseling-conversations-k",
                path=str(out_dir),
                unzip=True,
            )
            flag.touch()
            return out_dir
        except Exception as e:
            print(f"[DataLoader] Kaggle download failed: {e}")

    print("[DataLoader] Using built-in counseling sample…")
    _write_synthetic_counseling(out_dir)
    flag.touch()
    return out_dir


# ── Synthetic fallback data ───────────────────────────────────────────────────

def _write_synthetic_emotions(out_dir: Path):
    """Write a representative 3,000-sample emotions dataset."""
    samples = {
        "joy": [
            "I feel so happy and grateful today, everything is going well",
            "I am overjoyed, my dreams are finally coming true",
            "This is the best day I have had in a long time, pure joy",
            "I feel blessed and content with everything in my life",
            "I am excited about the future and feel motivated and hopeful",
            "Everything feels wonderful and I am in a great mood",
            "I feel a deep sense of peace and happiness right now",
            "Life is beautiful and I am grateful for all I have",
            "I am laughing and smiling, it feels so good to be happy",
            "I feel light and joyful, like nothing can bring me down",
        ] * 15,  # reduced from 40 to balance against negative classes
        "sadness": [
            "I feel so sad and hopeless, nothing seems to get better",
            "I am depressed and empty inside, I can't stop crying",
            "Everything feels pointless and I don't see the purpose anymore",
            "I am grieving and the pain feels unbearable right now",
            "I feel like I am drowning in sadness and cannot get out",
            "I miss the people I have lost and the pain won't go away",
            "I feel numb and disconnected from everything and everyone",
            "I am heartbroken and don't know how to move forward",
            "I feel worthless and like no one would miss me if I was gone",
            "The sadness is crushing me and I feel completely alone",
        ] * 40,
        "anger": [
            "I am so angry and frustrated, this is completely unfair",
            "I feel rage building inside me and I want to explode",
            "I am furious at the way I have been treated, it is unjust",
            "I cannot control my anger, everything is making me mad",
            "I feel bitter and resentful, the injustice makes me furious",
            "I am livid and feel like I could scream right now",
            "My frustration has turned into full blown rage and anger",
            "I feel hostile and aggressive, people keep pushing my buttons",
            "The anger is consuming me and I cannot calm down",
            "I am outraged by what happened, it was completely unacceptable",
        ] * 40,
        "fear": [
            "I am terrified and cannot shake this overwhelming fear",
            "I feel anxious and afraid, my heart is pounding with fear",
            "I am scared of what the future might hold for me",
            "I dread leaving the house and feel paralysed by fear",
            "I feel panicked and afraid, like something terrible will happen",
            "The fear is constant and I cannot find any peace or safety",
            "I am nervous and apprehensive about everything in my life",
            "I feel a deep sense of dread and cannot explain why I am scared",
            "I am frightened by my own thoughts and cannot stop them",
            "Anxiety and fear are controlling my life and I feel trapped",
        ] * 40,
        "love": [
            "I feel so much love and warmth for the people in my life",
            "I am deeply in love and it fills me with joy and tenderness",
            "I feel compassion and care for everyone around me today",
            "My heart is full of love and I feel deeply connected",
            "I love my family and friends so much it brings me to tears",
            "I feel grateful and loving towards everyone who supports me",
            "The love I feel is overwhelming in the best possible way",
            "I feel a deep bond with the people I care about most",
            "Love is keeping me going through the difficult times",
            "I feel warmth and affection for everyone in my life today",
        ] * 15,  # reduced from 40
        "surprise": [
            "I am completely shocked by what just happened, I did not expect this",
            "I feel surprised and amazed by the unexpected good news",
            "I cannot believe this happened, I am truly astonished",
            "I feel stunned and do not know how to process this news",
            "This was completely unexpected and I am still in disbelief",
            "I am pleasantly surprised by how things turned out today",
            "I feel shocked and surprised by the sudden change of events",
            "I did not see this coming at all, what a surprise",
            "I am amazed and taken aback by what just occurred",
            "This surprise has completely changed how I feel today",
        ] * 15,  # reduced from 40
    }

    rows = []
    for label, texts in samples.items():
        for text in texts:
            rows.append({"text": text, "label": label})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    split1 = int(len(df) * 0.7)
    split2 = int(len(df) * 0.85)

    def write_split(subset, fname):
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            for _, row in subset.iterrows():
                f.write(f"{row['text']};{row['label']}\n")

    write_split(df.iloc[:split1], "train.txt")
    write_split(df.iloc[split1:split2], "val.txt")
    write_split(df.iloc[split2:], "test.txt")
    print(f"[DataLoader] Synthetic emotions: {len(df)} samples written.")


def _write_synthetic_counseling(out_dir: Path):
    """Write a representative counseling conversations dataset."""
    convos = [
        ("I have been feeling extremely anxious lately and cannot stop worrying about everything. My mind races at night and I cannot sleep.",
         "Anxiety is a very common experience and what you are describing sounds like generalised anxiety. Try focusing on slow deep breathing when the thoughts race. Inhale for 4 counts, hold for 4, exhale for 6. This activates your parasympathetic nervous system and signals safety to your brain."),
        ("I feel so depressed and hopeless. Nothing brings me joy anymore and I do not see the point in anything.",
         "What you are describing sounds like depression, and I want you to know that it is a medical condition, not a personal failing. Behavioural activation is one of the most evidence-based approaches: even small actions like a short walk or calling a friend can begin to break the depression cycle."),
        ("I am completely burned out from work. I feel exhausted all the time and dread going in every day.",
         "Burnout is a real and serious condition with three dimensions: exhaustion, cynicism, and a sense of inefficacy. The most important first step is rest without guilt. Your body and mind need genuine recovery time, not just sleep."),
        ("I keep having panic attacks in public and now I avoid going out altogether.",
         "Panic attacks, while terrifying, are not dangerous. They are your fight-or-flight response misfiring. The most effective treatment is gradual exposure with support. Avoiding situations only strengthens the anxiety in the long run."),
        ("I have been really struggling with low self-esteem. I compare myself to others constantly.",
         "Social comparison is extremely common but damaging. Our brains evolved to compare ourselves to others, but social media has made this comparison constant and unfair. Try keeping a daily record of three specific things you did well, no matter how small."),
        ("I feel completely alone and like no one understands what I am going through.",
         "Loneliness is one of the most painful human experiences. It is important to distinguish between being alone and feeling lonely, as they are different things. Finding even one person who understands, whether a friend, therapist, or support group, can make a profound difference."),
        ("I cannot sleep no matter what I try. I lie awake for hours with my mind going.",
         "Sleep issues are often driven by hyperarousal, where your nervous system stays activated at night. Cognitive Behavioural Therapy for Insomnia, known as CBT-I, is the most effective non-medication treatment and works better than sleeping pills in the long term."),
        ("I have been feeling very angry lately and I take it out on people I love.",
         "Anger is a secondary emotion that often covers primary feelings like fear, hurt, or shame. When you feel anger rising, the physiological sigh, a double inhale through the nose followed by a long exhale, can interrupt the escalation cycle within seconds."),
        ("I have zero motivation and keep procrastinating on everything that matters.",
         "Low motivation often stems from executive function difficulties, depression, or fear of failure. The five-minute rule is scientifically supported: commit to doing the task for just five minutes. Starting is the hardest part and once begun most people continue."),
        ("I feel overwhelmed by everything I have to do and like I cannot cope.",
         "Overwhelm happens when demands exceed our perceived capacity. Brain dumping everything onto paper removes it from your working memory. Then identify one single most important task. You do not need to solve everything, only take one step."),
        ("I struggle with intrusive thoughts that I cannot control and they scare me.",
         "Intrusive thoughts are a universal human experience and the research is clear: having a thought does not make you a bad person. The problem is not the thought itself but the meaning you attach to it. Mindfulness teaches you to observe thoughts as passing events, not facts."),
        ("I feel like a failure at everything. I cannot seem to get anything right.",
         "The inner critic you are describing is often a protective mechanism that developed early in life. Cognitive defusion techniques from ACT therapy can help you see these thoughts as just words your mind produces, not truths about who you are."),
        ("My relationship is causing me so much stress and anxiety.",
         "Relationship stress is one of the leading causes of anxiety and depression. Attachment theory research shows that our earliest relationships shape how we connect as adults. Couples therapy or individual therapy focusing on attachment patterns can be transformative."),
        ("I feel disconnected from myself, like I am just going through the motions.",
         "What you are describing sounds like depersonalisation or emotional numbness, which often occurs as a protective response to stress or trauma. Grounding exercises, especially those involving the physical senses, can help restore the connection to your present experience."),
        ("I have been through a traumatic experience and cannot stop thinking about it.",
         "What you are experiencing sounds like trauma responses, which are normal reactions to abnormal events. Trauma-focused therapies like EMDR and Trauma-focused CBT have the strongest evidence base. Please consider reaching out to a trauma-informed therapist."),
    ] * 20  # repeat to create bigger dataset

    rows = []
    for q, a in convos:
        mood = label_mood_from_keywords(q)
        rows.append({
            "questionText": q,
            "answerText": a,
            "mood_label": mood,
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(out_dir / "Counseling_Data.csv", index=False)
    print(f"[DataLoader] Synthetic counseling: {len(df)} samples written.")


# ── Preprocessing ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, remove URLs, extra whitespace, and non-ASCII."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s\'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_emotions_df(split: str = "train") -> pd.DataFrame:
    """
    Load the emotions dataset split (train/val/test).
    Returns DataFrame with columns: text, label, mood, cleaned_text.
    """
    out_dir = download_emotions_dataset()

    fname_map = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}
    fpath = out_dir / fname_map.get(split, "train.txt")

    if not fpath.exists():
        # Some Kaggle versions use different filenames
        candidates = list(out_dir.glob("*.txt"))
        if candidates:
            fpath = candidates[0]

    rows = []
    with open(fpath, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: "text;label" or "text,label"
            sep = ";" if ";" in line else ","
            parts = line.rsplit(sep, 1)
            if len(parts) == 2:
                text, label = parts[0].strip(), parts[1].strip()
                rows.append({"text": text, "label": label})

    df = pd.DataFrame(rows)
    df["mood"] = df["label"].map(EMOTION_TO_MOOD).fillna("stress")
    df["cleaned_text"] = df["text"].apply(clean_text)
    return df


def load_counseling_df() -> pd.DataFrame:
    """
    Load the counseling conversations dataset.
    Returns DataFrame with mood labels derived from keyword analysis.
    """
    out_dir = download_counseling_dataset()

    # Try multiple filenames
    candidates = list(out_dir.glob("*.csv"))
    if not candidates:
        _write_synthetic_counseling(out_dir)
        candidates = list(out_dir.glob("*.csv"))

    df = pd.read_csv(candidates[0])

    # Normalise column names
    col_map = {}
    for col in df.columns:
        low = col.lower().strip()
        if "question" in low:
            col_map[col] = "questionText"
        elif "answer" in low or "response" in low:
            col_map[col] = "answerText"
    df = df.rename(columns=col_map)

    if "questionText" not in df.columns:
        df["questionText"] = df.iloc[:, 0].astype(str)
    if "answerText" not in df.columns:
        df["answerText"] = df.iloc[:, 1].astype(str) if df.shape[1] > 1 else ""

    # Label moods
    if "mood_label" not in df.columns:
        df["mood_label"] = df["questionText"].apply(label_mood_from_keywords)

    df["cleaned_question"] = df["questionText"].apply(clean_text)
    df["cleaned_answer"] = df["answerText"].apply(clean_text)
    return df


def build_unified_training_data() -> pd.DataFrame:
    """
    Merge emotions dataset + counseling dataset into one unified training set
    with columns: text, mood, source.
    Saves to processed/unified_training.csv.
    """
    cache = PROC_DIR / "unified_training.csv"
    if cache.exists():
        print("[DataLoader] Loading cached unified training data…")
        return pd.read_csv(cache)

    print("[DataLoader] Building unified training dataset…")

    # Emotions dataset (all splits merged)
    frames = []
    for split in ["train", "val", "test"]:
        try:
            frames.append(load_emotions_df(split))
        except Exception:
            pass
    emotions_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not emotions_df.empty:
        emotions_df = emotions_df[["cleaned_text", "mood"]].rename(columns={"cleaned_text": "text"})
        emotions_df["source"] = "emotions_kaggle"

    # Counseling dataset
    counseling_df = load_counseling_df()
    counseling_df = counseling_df[["cleaned_question", "mood_label"]].rename(
        columns={"cleaned_question": "text", "mood_label": "mood"}
    )
    counseling_df["source"] = "counseling_kaggle"

    # Merge
    # ── Add explicit training samples for extended mood classes ──
    extended_samples = {
        "stress": [
            "I am so stressed out I cannot think straight",
            "Work is stressing me beyond my limits",
            "I feel stressed and tense all the time",
            "The pressure is too much and I am stressed",
            "I am burned out and stressed from everything",
            "Stress is ruining my health and my life",
            "I feel frantic and stressed with no relief in sight",
            "I cannot relax because I am so stressed",
            "Every day is stressful and exhausting for me",
            "I am under so much stress I feel like breaking",
        ] * 30,
        "low_motivation": [
            "I have no motivation to do anything at all",
            "I feel unmotivated and stuck in life",
            "I keep procrastinating and cannot get started",
            "I have lost all drive and energy",
            "Nothing excites me anymore and I feel apathetic",
            "I cannot bring myself to do anything productive",
            "I feel lazy and have zero motivation",
            "I am stuck and have no desire to move forward",
            "Everything feels pointless so why even try",
            "I have no energy or willpower to accomplish anything",
        ] * 30,
        "loneliness": [
            "I feel so lonely and isolated from everyone",
            "Nobody understands me and I am all alone",
            "I have no friends and feel completely isolated",
            "I feel disconnected from everyone around me",
            "The loneliness is unbearable and painful",
            "I feel invisible and like nobody cares about me",
            "I am lonely and have no one to talk to",
            "I feel abandoned and rejected by everyone",
            "No one reaches out to me and I feel so alone",
            "I feel left out and excluded from everything",
        ] * 30,
        "insomnia": [
            "I cannot sleep at night no matter what I try",
            "I lie awake for hours and cannot fall asleep",
            "My insomnia is getting worse every night",
            "I wake up multiple times and cannot rest",
            "I am exhausted but my mind will not let me sleep",
            "Sleep has become impossible for me lately",
            "I dread going to bed because I know I will not sleep",
            "I have not had a good night of sleep in weeks",
            "My mind races at night and keeps me awake",
            "I feel tired all day because I cannot sleep at night",
        ] * 30,
        "overwhelmed": [
            "Everything is too much and I feel overwhelmed",
            "I am at my breaking point and cannot handle more",
            "I feel overwhelmed by all my responsibilities",
            "I am drowning in tasks and feel paralysed",
            "Everything is falling apart and I feel helpless",
            "I cannot cope with everything that is happening",
            "The weight of everything is crushing me",
            "I feel like I am sinking under all this pressure",
            "I have lost control and feel completely overwhelmed",
            "There is too much going on and I cannot handle it",
        ] * 30,
        "low_self_esteem": [
            "I feel worthless and not good enough",
            "I hate myself and think I am a failure",
            "I have no confidence in myself at all",
            "I constantly compare myself to others and feel inferior",
            "I feel ashamed of who I am",
            "I am not worthy of love or happiness",
            "I feel like everything I do is wrong",
            "I believe I am fundamentally broken and flawed",
            "My self esteem is so low I cannot face people",
            "I feel inadequate and useless compared to everyone",
        ] * 30,
    }
    ext_rows = []
    for mood, texts in extended_samples.items():
        for t in texts:
            ext_rows.append({"text": clean_text(t), "mood": mood, "source": "extended_synthetic"})
    extended_df = pd.DataFrame(ext_rows)

    unified = pd.concat([emotions_df, counseling_df, extended_df], ignore_index=True)
    unified = unified.dropna(subset=["text", "mood"])
    unified = unified[unified["text"].str.len() > 10]

    # ── Balance classes: cap the positive class ──
    max_per_class = unified[unified["mood"] != "positive"]["mood"].value_counts().max()
    balanced_parts = []
    for mood_label in unified["mood"].unique():
        subset = unified[unified["mood"] == mood_label]
        if len(subset) > max_per_class:
            subset = subset.sample(n=max_per_class, random_state=42)
        balanced_parts.append(subset)
    unified = pd.concat(balanced_parts, ignore_index=True)
    unified = unified.sample(frac=1, random_state=42).reset_index(drop=True)

    unified.to_csv(cache, index=False)
    print(f"[DataLoader] Unified dataset: {len(unified):,} samples, "
          f"{unified['mood'].nunique()} mood classes")
    print(unified["mood"].value_counts().to_string())
    return unified


if __name__ == "__main__":
    df = build_unified_training_data()
    print(df.head(10).to_string())
