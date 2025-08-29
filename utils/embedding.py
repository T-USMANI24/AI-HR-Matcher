#embeddings
from typing import List, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Optional Sentence-BERT fallback
try:
    from sentence_transformers import SentenceTransformer, util
    SBERT = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    SBERT = None

from utils.preprocess import clean_feedback

TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "20000"))

# --------------------------
# JD vs CV similarity
# --------------------------
def tfidf_similarity(jd_text: str, cv_texts: List[str]) -> List[float]:
    docs = [jd_text] + list(cv_texts)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=TFIDF_MAX_FEATURES)
    tfidf = vectorizer.fit_transform(docs)
    jd_vec = tfidf[0]
    cv_vecs = tfidf[1:]
    sims = cosine_similarity(cv_vecs, jd_vec).flatten().tolist()
    return [float(max(0.0, min(1.0, s))) for s in sims]

def sbert_similarity(jd_text: str, cv_texts: List[str]) -> List[float]:
    if SBERT is None:
        raise RuntimeError("Sentence-BERT model not available.")
    jd_emb = SBERT.encode(jd_text, convert_to_tensor=True, normalize_embeddings=True)
    cvs_emb = SBERT.encode(cv_texts, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(cvs_emb, jd_emb).cpu().numpy().flatten().clip(0,1).tolist()
    return [float(s) for s in sims]

def compute_similarity(cv_texts: List[str], jd_text: str, method: Optional[str] = None) -> List[float]:
    """
    method: None (auto: SBERT if available else TFIDF), "tfidf", or "sbert"
    Returns list of floats in [0,1] matching order of cv_texts.
    """
    if not cv_texts:
        return []
    if method is None:
        method = "sbert" if SBERT is not None else "tfidf"
    if method == "sbert":
        try:
            return sbert_similarity(jd_text, cv_texts)
        except Exception:
            return tfidf_similarity(jd_text, cv_texts)
    else:
        return tfidf_similarity(jd_text, cv_texts)

# --------------------------
# Recruiter feedback embedding
# --------------------------
def embed_feedback(feedback_text: Union[str, List[str]], method: Optional[str] = None):
    """
    Clean and embed recruiter feedback (single string or list of strings).
    Returns embedding(s) or None if input empty.
    """
    if not feedback_text:
        return None

    # normalize input to list
    if isinstance(feedback_text, str):
        texts = [clean_feedback(feedback_text)]
    else:
        texts = [clean_feedback(t) for t in feedback_text]

    if method is None:
        method = "sbert" if SBERT is not None else "tfidf"

    if method == "sbert" and SBERT is not None:
        return SBERT.encode(texts, normalize_embeddings=True)

    # fallback TF-IDF embedding (not similarity, raw vector)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=TFIDF_MAX_FEATURES)
    vecs = vectorizer.fit_transform(texts)
    return vecs.toarray()
