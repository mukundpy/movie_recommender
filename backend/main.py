import os
import pickle
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# =========================
# ENV
# =========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE    = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError(
        "TMDB_API_KEY missing. Create a .env file in backend/ with:\n"
        "TMDB_API_KEY=your_key_here\n"
        "Get a free key at https://www.themoviedb.org/settings/api"
    )

# =========================
# FASTAPI APP
# =========================
# Lifespan is wired below after load_pickles is defined
app = FastAPI(title="Movie Recommender API", version="3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PICKLE GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH          = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH     = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH= os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH       = os.path.join(BASE_DIR, "tfidf.pkl")

df:           Optional[pd.DataFrame] = None
indices_obj:  Any                    = None
tfidf_matrix: Any                    = None
tfidf_obj:    Any                    = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None

# =========================
# MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id:      int
    title:        str
    poster_url:   Optional[str]   = None
    release_date: Optional[str]   = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id:      int
    title:        str
    overview:     Optional[str]  = None
    release_date: Optional[str]  = None
    poster_url:   Optional[str]  = None
    backdrop_url: Optional[str]  = None
    genres:       List[dict]     = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb:  Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query:                  str
    movie_details:          TMDBMovieDetails
    tfidf_recommendations:  List[TFIDFRecItem]
    genre_recommendations:  List[TMDBMovieCard]


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(params)
    q["api_key"] = TMDB_API_KEY
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"TMDB request error: {repr(e)}")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"TMDB error {r.status_code}: {r.text}")
    return r.json()


async def tmdb_cards_from_results(results: List[dict], limit: int = 20) -> List[TMDBMovieCard]:
    return [
        TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or m.get("name") or "",
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
        for m in (results or [])[:limit]
    ]


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return await tmdb_get(
        "/search/movie",
        {"query": query, "include_adult": "false", "language": "en-US", "page": page},
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


# =========================
# TF-IDF HELPERS
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    Accepts dict or pandas Series (from indices.pkl).
    Returns normalized lowercase title -> int index map.
    Deduplicates: first occurrence wins (matches notebook fix).
    """
    title_to_idx: Dict[str, int] = {}
    try:
        for k, v in indices.items():
            nk = _norm_title(k)
            if nk not in title_to_idx:          # keep first occurrence only
                title_to_idx[nk] = int(v)
    except Exception as e:
        raise RuntimeError(f"indices.pkl must support .items(): {e}")
    return title_to_idx


def get_local_idx_by_title(title: str) -> int:
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(status_code=404, detail=f"Title not found in local dataset: '{title}'")


def tfidf_recommend_titles(query_title: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Returns [(title, cosine_score), ...] from local df.
    Uses sparse dot-product (fast, avoids dense cosine_similarity on full matrix).
    Re-ranks with vote_average and popularity to filter out low-quality results.
    """
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx    = get_local_idx_by_title(query_title)
    qv     = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    # Fetch more candidates than needed so re-ranking has room to work
    candidate_n = top_n * 3
    order = np.argsort(-scores)

    candidates = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            row         = df.iloc[int(i)]
            title_i     = str(row["title"])
            vote_avg    = float(row.get("vote_average", 0) or 0)
            popularity  = float(row.get("popularity", 0) or 0)
        except Exception:
            continue
        candidates.append((title_i, float(scores[int(i)]), vote_avg, popularity))
        if len(candidates) >= candidate_n:
            break

    # Normalize vote and popularity for re-ranking
    if candidates:
        pop_max  = df["popularity"].max() or 1.0
        reranked = []
        for title_i, sim, vote, pop in candidates:
            final = 0.60 * sim + 0.30 * (vote / 10.0) + 0.10 * (pop / pop_max)
            reranked.append((title_i, sim, final))
        reranked.sort(key=lambda x: x[2], reverse=True)
        return [(t, s) for t, s, _ in reranked[:top_n]]

    return []


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None


# =========================
# STARTUP: LOAD PICKLES
# =========================
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    missing = [p for p in [DF_PATH, INDICES_PATH, TFIDF_MATRIX_PATH, TFIDF_PATH]
               if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            f"Missing pickle files in backend/: {[os.path.basename(p) for p in missing]}\n"
            "Run the notebook and copy the 4 .pkl files into backend/"
        )

    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must be a DataFrame with a 'title' column")

    print(f"✅ Loaded {len(df)} movies | {len(TITLE_TO_IDX)} unique titles indexed")


@asynccontextmanager
async def lifespan(application):
    """Modern FastAPI lifespan: load resources on startup, cleanup on shutdown."""
    load_pickles()
    yield


# Wire lifespan into the app
app.router.lifespan_context = lifespan


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "movies_loaded": len(df) if df is not None else 0,
        "titles_indexed": len(TITLE_TO_IDX) if TITLE_TO_IDX else 0,
    }


@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    """
    Home feed.
    category: trending | popular | top_rated | upcoming | now_playing
    """
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)
        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")
        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page:  int  = Query(1, ge=1, le=10),
):
    """Keyword search — returns raw TMDB shape with 'results' list."""
    return await tmdb_search_movies(query=query, page=page)


@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit:   int = Query(18, ge=1, le=50),
):
    details  = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []
    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {"with_genres": genre_id, "language": "en-US", "sort_by": "popularity.desc", "page": 1},
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    """Debug endpoint — TF-IDF recommendations without TMDB enrichment."""
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": round(s, 4)} for t, s in recs]


@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query:        str = Query(..., min_length=1),
    tfidf_top_n:  int = Query(12, ge=1, le=30),
    genre_limit:  int = Query(12, ge=1, le=30),
):
    """
    Full bundle for the details page:
    - Best TMDB match for query
    - Movie details
    - TF-IDF recommendations (local model, enriched with TMDB posters)
    - Genre recommendations (TMDB discover)
    """
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(status_code=404, detail=f"No TMDB movie found for: '{query}'")

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    # ---- TF-IDF recs ----
    tfidf_items: List[TFIDFRecItem] = []
    recs: List[Tuple[str, float]] = []
    try:
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except HTTPException:
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []
    except Exception:
        recs = []

    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=round(score, 4), tmdb=card))

    # ---- Genre recs ----
    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {"with_genres": genre_id, "language": "en-US", "sort_by": "popularity.desc", "page": 1},
        )
        cards      = await tmdb_cards_from_results(discover.get("results", []), limit=genre_limit)
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )
