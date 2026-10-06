
import html
import re
import unicodedata
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rapidfuzz import fuzz


# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"

MODEL_PATH = MODEL_DIR / "xgboost_tuned.joblib"
VECTORIZER_PATH = MODEL_DIR / "word_tfidf_vectorizer.joblib"
METADATA_PATH = MODEL_DIR / "feature_metadata.joblib"


# =========================================================
# 2. EXPECTED FEATURE NAMES
# =========================================================

EXPECTED_FEATURES = [
    "q1_char_count",
    "q2_char_count",
    "char_count_diff",
    "q1_word_count",
    "q2_word_count",
    "word_count_diff",
    "char_length_ratio",
    "word_length_ratio",
    "common_word_count",
    "unique_word_count_q1",
    "unique_word_count_q2",
    "word_jaccard_similarity",
    "common_word_ratio_q1",
    "common_word_ratio_q2",
    "fuzzy_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "word_tfidf_similarity",
]


# =========================================================
# 3. TEXT PREPROCESSING
# Keep this consistent with 02_preprocessing.ipynb
# =========================================================

CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "it's": "it is",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "what's": "what is",
    "who's": "who is",
    "where's": "where is",
    "there's": "there is",
    "i've": "i have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "we'd": "we would",
}

CONTRACTION_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(key)
        for key in sorted(CONTRACTIONS, key=len, reverse=True)
    )
    + r")\b",
    flags=re.IGNORECASE,
)


def normalize_text(text):
    """Match the normalization used in the training notebook."""

    if not isinstance(text, str):
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Decode HTML entities
    text = html.unescape(text)

    # Lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Expand contractions
    text = CONTRACTION_PATTERN.sub(
        lambda match: CONTRACTIONS[match.group().lower()],
        text,
    )

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove non-printable control characters
    text = "".join(
        char for char in text
        if char.isprintable() or char.isspace()
    )

    # Final whitespace normalization
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# 4. LOAD TRAINED ARTIFACTS
# =========================================================

@lru_cache(maxsize=1)
def load_artifacts():
    """Load the existing trained model and supporting artifacts."""

    for path in (MODEL_PATH, VECTORIZER_PATH, METADATA_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required model artifact not found: {path}"
            )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    metadata = joblib.load(METADATA_PATH)

    if not isinstance(metadata, dict):
        raise RuntimeError("Invalid feature metadata format.")

    feature_names = metadata.get("feature_names")

    if not isinstance(feature_names, list):
        raise RuntimeError(
            "feature_names is missing from feature_metadata.joblib."
        )

    if feature_names != EXPECTED_FEATURES:
        raise RuntimeError(
            "Feature metadata does not match the training pipeline.\n"
            f"Expected: {EXPECTED_FEATURES}\n"
            f"Received: {feature_names}"
        )

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "The loaded model does not support predict_proba."
        )

    if not hasattr(vectorizer, "vocabulary_"):
        raise RuntimeError("The TF-IDF vectorizer is not fitted.")

    # Validate the model's feature names if available.
    model_features = getattr(model, "feature_names_in_", None)

    if model_features is not None:
        if list(model_features) != feature_names:
            raise RuntimeError(
                "Model feature order differs from metadata."
            )

    if getattr(model, "n_features_in_", len(feature_names)) != len(
        feature_names
    ):
        raise RuntimeError(
            "Model feature count does not match metadata."
        )

    return model, vectorizer, metadata


# =========================================================
# 5. HANDCRAFTED FEATURE ENGINEERING
# Matches 04_model_training.ipynb
# =========================================================

def extract_pair_features(q1, q2):
    """Extract the 17 handcrafted features."""

    words1 = set(q1.split())
    words2 = set(q2.split())

    common_words = words1 & words2
    total_words = words1 | words2

    q1_word_count = len(q1.split())
    q2_word_count = len(q2.split())

    q1_char_count = len(q1)
    q2_char_count = len(q2)

    return {
        "q1_char_count": q1_char_count,
        "q2_char_count": q2_char_count,

        "char_count_diff": abs(
            q1_char_count - q2_char_count
        ),

        "q1_word_count": q1_word_count,
        "q2_word_count": q2_word_count,

        "word_count_diff": abs(
            q1_word_count - q2_word_count
        ),

        "char_length_ratio": (
            min(q1_char_count, q2_char_count)
            / max(q1_char_count, q2_char_count)
            if max(q1_char_count, q2_char_count)
            else 0.0
        ),

        "word_length_ratio": (
            min(q1_word_count, q2_word_count)
            / max(q1_word_count, q2_word_count)
            if max(q1_word_count, q2_word_count)
            else 0.0
        ),

        "common_word_count": len(common_words),

        "unique_word_count_q1": len(words1),
        "unique_word_count_q2": len(words2),

        "word_jaccard_similarity": (
            len(common_words) / len(total_words)
            if total_words
            else 0.0
        ),

        "common_word_ratio_q1": (
            len(common_words) / len(words1)
            if words1
            else 0.0
        ),

        "common_word_ratio_q2": (
            len(common_words) / len(words2)
            if words2
            else 0.0
        ),

        "fuzzy_ratio": fuzz.ratio(q1, q2),
        "token_sort_ratio": fuzz.token_sort_ratio(q1, q2),
        "token_set_ratio": fuzz.token_set_ratio(q1, q2),
    }


# =========================================================
# 6. COMPLETE INFERENCE FEATURE PIPELINE
# =========================================================

def extract_features(question1, question2, vectorizer, metadata):
    """Generate all 18 features in the exact training order."""

    # Apply the same preprocessing used during training.
    q1 = normalize_text(question1)
    q2 = normalize_text(question2)

    # Extract the 17 handcrafted features.
    features = extract_pair_features(q1, q2)

    # Generate TF-IDF representations using the fitted vectorizer.
    q1_vector = vectorizer.transform([q1])
    q2_vector = vectorizer.transform([q2])

    # Cosine similarity; the fitted vectorizer uses L2 normalization.
    word_tfidf_similarity = float(
        q1_vector.multiply(q2_vector).sum()
    )

    features["word_tfidf_similarity"] = word_tfidf_similarity

    # Preserve the precise feature order from training.
    feature_names = metadata["feature_names"]

    missing = [
        name for name in feature_names
        if name not in features
    ]

    if missing:
        raise RuntimeError(
            f"Missing inference features: {missing}"
        )

    # Keep the same numeric data type used for feature generation.
    feature_df = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names,
        dtype=np.float32,
    )

    if not np.isfinite(feature_df.to_numpy()).all():
        raise RuntimeError(
            "Non-finite values found in inference features."
        )

    return feature_df


# =========================================================
# 7. PREDICTION
# =========================================================

def predict_similarity(question1, question2, threshold=0.5):
    """Predict duplicate probability for a pair of questions."""

    if not question1.strip() or not question2.strip():
        raise ValueError("Both questions must contain text.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    model, vectorizer, metadata = load_artifacts()

    features = extract_features(
        question1,
        question2,
        vectorizer,
        metadata,
    )

    probability = float(model.predict_proba(features)[0, 1])

    return {
        "duplicate_probability": round(probability, 6),
        "similarity_percentage": round(probability * 100, 2),
        "is_duplicate": bool(probability >= threshold),
        "threshold": float(threshold),
    }


# =========================================================
# 8. FASTAPI APPLICATION
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load artifacts and validate the pipeline on startup.
    load_artifacts()
    print("QuerySense model and artifacts loaded successfully.")
    yield


app = FastAPI(
    title="QuerySense API",
    description="Quora Question Similarity Prediction",
    version="1.1.0",
    lifespan=lifespan,
)


# =========================================================
# 9. REQUEST SCHEMA
# =========================================================

class PredictionRequest(BaseModel):
    question1: str = Field(min_length=1, max_length=2000)
    question2: str = Field(min_length=1, max_length=2000)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


# =========================================================
# 10. API ENDPOINTS
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "XGBoost",
        "features": len(EXPECTED_FEATURES),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if not request.question1.strip() or not request.question2.strip():
        raise HTTPException(
            status_code=422,
            detail="Both questions must contain text.",
        )

    try:
        return predict_similarity(
            question1=request.question1,
            question2=request.question2,
            threshold=request.threshold,
        )

    except Exception as error:
        print(f"Prediction error: {error}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend terminal.",
        ) from error


# Ignore a browser's optional favicon request.
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


# =========================================================
# 11. SERVE THE FRONTEND
# =========================================================

if not FRONTEND_DIR.is_dir():
    raise RuntimeError(
        f"Frontend directory not found: {FRONTEND_DIR}"
    )

if not (FRONTEND_DIR / "index.html").is_file():
    raise RuntimeError(
        f"index.html not found inside: {FRONTEND_DIR}"
    )

# Mount last so API routes are registered before static files.
app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend",
)


# =========================================================
# 12. RUN THE APPLICATION
# =========================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )