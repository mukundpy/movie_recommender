import requests
import streamlit as st

# =============================
# CONFIG
# =============================
# Change to your deployed URL or keep localhost for local dev
API_BASE = "https://movie-recommender-vq66.onrender.com"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# =============================
# STYLES
# =============================
st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
.small-muted { color: #6b7280; font-size: 0.92rem; }
.movie-title { font-size: 0.9rem; line-height: 1.15rem; height: 2.3rem; overflow: hidden; }
.card { border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; padding: 14px; background: rgba(255,255,255,0.7); }
.section-header { font-size: 1.1rem; font-weight: 600; margin-top: 1rem; margin-bottom: 0.4rem; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# SESSION STATE + ROUTING
# =============================
if "view" not in st.session_state:
    st.session_state.view = "home"
if "selected_tmdb_id" not in st.session_state:
    st.session_state.selected_tmdb_id = None

# Sync from query params (supports browser back/forward)
qp_view = st.query_params.get("view")
qp_id   = st.query_params.get("id")
if qp_view in ("home", "details"):
    st.session_state.view = qp_view
if qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except Exception:
        pass


def goto_home():
    st.session_state.view = "home"
    st.query_params["view"] = "home"
    if "id" in st.query_params:
        del st.query_params["id"]
    st.rerun()


def goto_details(tmdb_id: int):
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)
    st.query_params["view"] = "details"
    st.query_params["id"]   = str(int(tmdb_id))
    st.rerun()


# =============================
# API HELPERS
# =============================
@st.cache_data(ttl=60)
def api_get_json(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=25)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
        return r.json(), None
    except Exception as e:
        return None, f"Request failed: {e}"


def poster_grid(cards, cols=6, key_prefix="grid"):
    """Render a grid of movie poster cards with an Open button each."""
    if not cards:
        st.info("No movies to show.")
        return

    rows = (len(cards) + cols - 1) // cols
    idx  = 0
    for r in range(rows):
        colset = st.columns(cols)
        for c in range(cols):
            if idx >= len(cards):
                break
            m       = cards[idx]
            idx    += 1
            tmdb_id = m.get("tmdb_id")
            title   = m.get("title", "Untitled")
            poster  = m.get("poster_url")

            with colset[c]:
                if poster:
                    st.image(poster, use_column_width=True)
                else:
                    st.markdown("🖼️ No poster")

                if tmdb_id and st.button("Open", key=f"{key_prefix}_{r}_{c}_{idx}_{tmdb_id}"):
                    goto_details(tmdb_id)

                st.markdown(f"<div class='movie-title'>{title}</div>", unsafe_allow_html=True)


def to_cards_from_tfidf_items(tfidf_items):
    """Convert TF-IDF rec items (with nested tmdb field) to flat card dicts."""
    cards = []
    for x in tfidf_items or []:
        tmdb = x.get("tmdb") or {}
        if tmdb.get("tmdb_id"):
            cards.append(
                {
                    "tmdb_id":    tmdb["tmdb_id"],
                    "title":      tmdb.get("title") or x.get("title") or "Untitled",
                    "poster_url": tmdb.get("poster_url"),
                }
            )
    return cards


def parse_tmdb_search_to_cards(data, keyword: str, limit: int = 24):
    """
    Handles both API response shapes:
    - Raw TMDB: {"results": [{id, title, poster_path, ...}]}
    - List:     [{tmdb_id, title, poster_url, ...}]

    Returns:
      suggestions : list[(label, tmdb_id)]
      cards       : list[{tmdb_id, title, poster_url}]
    """
    keyword_l = keyword.strip().lower()

    if isinstance(data, dict) and "results" in data:
        raw_items = []
        for m in data.get("results") or []:
            title   = (m.get("title") or "").strip()
            tmdb_id = m.get("id")
            if not title or not tmdb_id:
                continue
            poster_path = m.get("poster_path")
            raw_items.append(
                {
                    "tmdb_id":      int(tmdb_id),
                    "title":        title,
                    "poster_url":   f"{TMDB_IMG}{poster_path}" if poster_path else None,
                    "release_date": m.get("release_date", ""),
                }
            )
    elif isinstance(data, list):
        raw_items = []
        for m in data:
            tmdb_id = m.get("tmdb_id") or m.get("id")
            title   = (m.get("title") or "").strip()
            if not title or not tmdb_id:
                continue
            raw_items.append(
                {
                    "tmdb_id":      int(tmdb_id),
                    "title":        title,
                    "poster_url":   m.get("poster_url"),
                    "release_date": m.get("release_date", ""),
                }
            )
    else:
        return [], []

    matched    = [x for x in raw_items if keyword_l in x["title"].lower()]
    final_list = matched if matched else raw_items

    suggestions = []
    for x in final_list[:10]:
        year  = (x.get("release_date") or "")[:4]
        label = f"{x['title']} ({year})" if year else x["title"]
        suggestions.append((label, x["tmdb_id"]))

    cards = [
        {"tmdb_id": x["tmdb_id"], "title": x["title"], "poster_url": x["poster_url"]}
        for x in final_list[:limit]
    ]
    return suggestions, cards


# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.markdown("## 🎬 Menu")
    if st.button("🏠 Home"):
        goto_home()

    st.markdown("---")
    st.markdown("### Home Feed")
    home_category = st.selectbox(
        "Category",
        ["trending", "popular", "top_rated", "now_playing", "upcoming"],
        index=0,
    )
    grid_cols = st.slider("Grid columns", 4, 8, 6)

    st.markdown("---")

    # Health check widget
    if st.button("🔍 Check API health"):
        health, err = api_get_json("/health")
        if err:
            st.error(f"API offline: {err}")
        else:
            st.success(
                f"API OK ✅\n\n"
                f"Movies: {health.get('movies_loaded', '?')}\n\n"
                f"Indexed: {health.get('titles_indexed', '?')}"
            )

# =============================
# HEADER
# =============================
st.title("🎬 Movie Recommender")
st.markdown(
    "<div class='small-muted'>Search → select → get recommendations powered by TF-IDF + TMDB</div>",
    unsafe_allow_html=True,
)
st.divider()

# ==========================================================
# VIEW: HOME
# ==========================================================
if st.session_state.view == "home":
    typed = st.text_input(
        "🔍 Search by movie title", placeholder="e.g. avenger, batman, inception..."
    )

    st.divider()

    # ---- SEARCH MODE ----
    if typed.strip():
        if len(typed.strip()) < 2:
            st.caption("Type at least 2 characters.")
        else:
            with st.spinner("Searching..."):
                data, err = api_get_json("/tmdb/search", params={"query": typed.strip()})

            if err or data is None:
                st.error(f"Search failed: {err}")
            else:
                suggestions, cards = parse_tmdb_search_to_cards(data, typed.strip(), limit=24)

                if suggestions:
                    labels   = ["-- Select a movie --"] + [s[0] for s in suggestions]
                    selected = st.selectbox("Suggestions (select to open details)", labels, index=0)

                    if selected != "-- Select a movie --":
                        label_to_id = {s[0]: s[1] for s in suggestions}
                        goto_details(label_to_id[selected])
                else:
                    st.info("No suggestions found. Try another keyword.")

                st.markdown("### 🔎 Search Results")
                poster_grid(cards, cols=grid_cols, key_prefix="search_results")

        st.stop()

    # ---- HOME FEED MODE ----
    st.markdown(f"### 🏠 {home_category.replace('_', ' ').title()}")

    with st.spinner("Loading..."):
        home_cards, err = api_get_json("/home", params={"category": home_category, "limit": 24})

    if err or not home_cards:
        st.error(f"Home feed failed: {err or 'Unknown error'}")
        st.stop()

    poster_grid(home_cards, cols=grid_cols, key_prefix="home_feed")


# ==========================================================
# VIEW: DETAILS
# ==========================================================
elif st.session_state.view == "details":
    tmdb_id = st.session_state.selected_tmdb_id
    if not tmdb_id:
        st.warning("No movie selected.")
        if st.button("← Back to Home"):
            goto_home()
        st.stop()

    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown("### 📄 Movie Details")
    with col_b:
        if st.button("← Back to Home"):
            goto_home()

    with st.spinner("Loading movie details..."):
        data, err = api_get_json(f"/movie/id/{tmdb_id}")

    if err or not data:
        st.error(f"Could not load details: {err or 'Unknown error'}")
        st.stop()

    # ---- Layout: Poster | Details ----
    left, right = st.columns([1, 2.4], gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if data.get("poster_url"):
            st.image(data["poster_url"], use_column_width=True)
        else:
            st.write("🖼️ No poster")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"## {data.get('title', '')}")
        release = data.get("release_date") or "—"
        genres  = ", ".join([g["name"] for g in data.get("genres", [])]) or "—"
        rating  = data.get("vote_average")
        st.markdown(f"<div class='small-muted'>📅 Release: {release}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>🎭 Genres: {genres}</div>",   unsafe_allow_html=True)
        if rating:
            st.markdown(f"<div class='small-muted'>⭐ Rating: {rating}/10</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Overview")
        st.write(data.get("overview") or "No overview available.")
        st.markdown("</div>", unsafe_allow_html=True)

    if data.get("backdrop_url"):
        st.markdown("#### Backdrop")
        st.image(data["backdrop_url"], use_column_width=True)

    st.divider()
    st.markdown("### 🎯 Recommendations")

    title = (data.get("title") or "").strip()
    if title:
        with st.spinner("Loading recommendations..."):
            bundle, err2 = api_get_json(
                "/movie/search",
                params={"query": title, "tfidf_top_n": 12, "genre_limit": 12},
            )

        if not err2 and bundle:
            tfidf_cards = to_cards_from_tfidf_items(bundle.get("tfidf_recommendations"))
            genre_cards = bundle.get("genre_recommendations", [])

            if tfidf_cards:
                st.markdown("#### 🔎 Similar Movies (TF-IDF)")
                poster_grid(tfidf_cards, cols=grid_cols, key_prefix="details_tfidf")
            else:
                st.info("No TF-IDF recommendations found (title may not be in local dataset).")

            if genre_cards:
                st.markdown("#### 🎭 More in Same Genre")
                poster_grid(genre_cards, cols=grid_cols, key_prefix="details_genre")

        else:
            # Fallback: genre-only recommendations
            st.info("Showing genre-based recommendations (TF-IDF fallback).")
            with st.spinner("Loading genre recommendations..."):
                genre_only, err3 = api_get_json("/recommend/genre", params={"tmdb_id": tmdb_id, "limit": 18})
            if not err3 and genre_only:
                poster_grid(genre_only, cols=grid_cols, key_prefix="details_genre_fallback")
            else:
                st.warning("No recommendations available right now.")
    else:
        st.warning("No title available to compute recommendations.")
