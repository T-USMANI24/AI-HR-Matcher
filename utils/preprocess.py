# utils/preprocess.py
import io
import os
import re
import fitz  # PyMuPDF
from PIL import Image

# Optional: pytesseract for OCR fallback (install if you want OCR)
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# --------------------------
# PDF bytes -> text (fast via PyMuPDF, optional OCR fallback)
# --------------------------
def _pdf_bytes_to_text(b: bytes) -> str:
    text = ""
    try:
        doc = fitz.open(stream=b, filetype="pdf")
        for page in doc:
            page_text = page.get_text() or ""
            text += page_text
            # fallback OCR on page if page_text empty and OCR available
            if not page_text.strip() and TESSERACT_AVAILABLE:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img)
        doc.close()
    except Exception:
        try:
            return b.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    return text

# --------------------------
# Flexible loader for path / file-like / bytes
# --------------------------
def prepare_document(file_or_bytes):
    """
    Accepts:
      - file path (str)
      - file-like with .read (Streamlit UploadedFile)
      - raw bytes
      - plain text (str) -> returned as-is (cleaned)
    Returns: raw extracted text (not heavily cleaned)
    """
    if file_or_bytes is None:
        return ""

    # file-like (Streamlit, Flask file storage etc.)
    if hasattr(file_or_bytes, "read"):
        name = getattr(file_or_bytes, "name", "")
        raw = file_or_bytes.read()
        try:
            file_or_bytes.seek(0)
        except Exception:
            pass
        if isinstance(raw, (bytes, bytearray)):
            if name.lower().endswith(".pdf"):
                return _pdf_bytes_to_text(raw)
            try:
                return raw.decode("utf-8", errors="ignore")
            except Exception:
                return str(raw)
        else:
            return str(raw)

    # string: might be a path or plain text
    if isinstance(file_or_bytes, str):
        # path
        if os.path.exists(file_or_bytes):
            if file_or_bytes.lower().endswith(".pdf"):
                with open(file_or_bytes, "rb") as f:
                    return _pdf_bytes_to_text(f.read())
            else:
                with open(file_or_bytes, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
        # otherwise treat as plain text
        return file_or_bytes

    # bytes
    if isinstance(file_or_bytes, (bytes, bytearray)):
        return _pdf_bytes_to_text(bytes(file_or_bytes))

    return ""

# --------------------------
# Basic cleaning for embeddings / TF-IDF
# --------------------------
def clean_text_for_model(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ascii noisy chars
    text = re.sub(r"[^\w\s\.\,]", " ", text)  # keep punctuation a bit
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --------------------------
# Simple skill and degree extraction (Day 1)
# --------------------------
# You can extend BASE_SKILLS later or load from a CSV/DB
BASE_SKILLS = [
    "python","sql","java","excel","powerpoint","word","machine learning",
    "deep learning","data science","nlp","aws","azure","docker","kubernetes",
    "project management","communication","sales","photoshop","illustrator","react","flask"
]

SKILL_SYNONYMS = {
    "excel": ["ms excel", "microsoft excel"],
    "word": ["ms word", "microsoft word"],
    "powerpoint": ["ppt", "ms powerpoint", "microsoft powerpoint"],
    "machine learning": ["ml"],
    "deep learning": ["dl"],
    "nlp": ["natural language processing"]
}

DEGREE_PATTERNS = {
    "BTECH": ["b.tech","btech","b.e","b e","bachelor of technology","bachelor of engineering"],
    "MTECH": ["m.tech","mtech","m.e","master of technology","master of engineering"],
    "BSC": ["b.sc","bsc","bachelor of science"],
    "MSC": ["m.sc","msc","master of science"],
    "MBA": ["mba","master of business administration"],
    "PHD": ["ph.d","phd","doctor of philosophy"]
}

def extract_skills(text: str):
    """Find skills in text using simple substring checks + synonyms (case-insensitive)."""
    if not text:
        return []
    t = text.lower()
    found = set()
    for s in BASE_SKILLS:
        if re.search(rf"\b{re.escape(s)}\b", t):
            found.add(s)
        elif s in SKILL_SYNONYMS:
            for syn in SKILL_SYNONYMS[s]:
                if re.search(rf"\b{re.escape(syn)}\b", t):
                    found.add(s)
                    break
    return sorted(found)

def extract_degrees(text: str):
    if not text:
        return []
    t = text.lower()
    found = set()
    for key, pats in DEGREE_PATTERNS.items():
        for p in pats:
            if p in t:
                found.add(key)
                break
    return sorted(found)

# --------------------------
# Quick years-of-experience extractor (simple heuristic)
# --------------------------
def extract_experience_years(text: str):
    """
    Finds numeric patterns like '3 years', '5+ years', '2 yrs', etc.
    Returns max found or 0.
    """
    if not text:
        return 0.0
    t = text.lower()
    years = re.findall(r"(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years|yrs|year)", t)
    months = re.findall(r"(\d+(?:\.\d+)?)\s*(?:months|mos|month)", t)
    total = 0.0
    if years:
        total += max(float(y) for y in years)
    if months:
        total += max(float(m) for m in months)/12.0
    return round(total, 1)
# --------------------------
# Recruiter Feedback Preprocessing
# --------------------------
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

def clean_feedback(text: str) -> str:
    """
    Preprocess recruiter feedback text for sentiment/embedding models.
    Steps:
      - lowercase
      - remove special chars/numbers
      - lemmatization
      - remove stopwords/punctuations
    """
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)   # keep only alphabets + spaces
    
    if nlp:  # if spacy model available
        doc = nlp(text)
        tokens = [
            token.lemma_ for token in doc 
            if not token.is_stop and not token.is_punct
        ]
        return " ".join(tokens)
    
    # fallback: just collapse whitespace
    return re.sub(r'\s+', ' ', text).strip()
