"""Central settings, read from the environment with sensible defaults."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://vedb@localhost:5432/lighthouse"
)

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))
TOP_K = int(os.environ.get("TOP_K", "8"))

EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "local")

# bge-base-en-v1.5 is asymmetric and tuned for retrieval. The cache file name embeds
# the model name (see EMBED_CACHE_PATH below), so switching models uses a fresh cache.
LOCAL_EMBEDDING_MODEL = os.environ.get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST") or None
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2000"))
# Temperature 0 makes generation and judging deterministic.
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "briefs")

# bge-v1.5 is asymmetric: the QUERY must be prefixed with this instruction while
# documents are embedded as-is. Applied only in VectorStore.search.
BGE_QUERY_PREFIX = os.environ.get(
    "BGE_QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: ",
)

# Email domain treated as "internal" when classifying attendees (done in Python).
INTERNAL_EMAIL_DOMAIN = os.environ.get("INTERNAL_EMAIL_DOMAIN", "fastcode.ai")

# Recency weighting for retrieval. Each candidate chunk's cosine score is multiplied
# by 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS); set <= 0 to disable decay.
# CANDIDATE_MULTIPLIER widens the FAISS pool we re-rank so recent-but-slightly-less-
# similar chunks can win.
RECENCY_HALF_LIFE_DAYS = float(os.environ.get("RECENCY_HALF_LIFE_DAYS", "30"))
CANDIDATE_MULTIPLIER = int(os.environ.get("CANDIDATE_MULTIPLIER", "5"))

# Eval score = EVAL_GUIDELINE_WEIGHT * guideline_score
#            + EVAL_QUALITY_WEIGHT  * quality_score.
EVAL_GUIDELINE_WEIGHT = float(os.environ.get("EVAL_GUIDELINE_WEIGHT", "0.6"))
EVAL_QUALITY_WEIGHT = float(os.environ.get("EVAL_QUALITY_WEIGHT", "0.4"))
# Where eval scorecards are written.
EVAL_DIR = os.environ.get("EVAL_DIR", "eval")

# Cache
CACHE_DIR = os.environ.get("CACHE_DIR", ".cache")
# One cache file per embedding model: a different model produces different vectors,
# so putting the model name in the path invalidates the cache automatically.
EMBED_CACHE_PATH = os.path.join(
    CACHE_DIR, f"embeddings-{LOCAL_EMBEDDING_MODEL.replace('/', '_')}.pkl"
)