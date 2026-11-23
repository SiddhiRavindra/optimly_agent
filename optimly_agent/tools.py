"""Tools that read the public Optimly Index (no credentials needed).

Endpoints (documented in https://optimly.ai/llms.txt):
  - GET https://optimly.ai/brands/search?q=<name or domain>  -> candidate brands
  - GET https://optimly.ai/brand/<slug>.json                  -> one brand profile

Every HTTP response is cached to disk (.cache/) so evaluation runs are fast,
repeatable, and polite to Optimly's servers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
from langchain_core.tools import tool
from rapidfuzz import fuzz

BASE_URL = "https://optimly.ai"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
USER_AGENT = "optimly-agent-lab/0.1 (research demo)"
TIMEOUT = 20.0


# --------------------------------------------------------------------------- #
# HTTP + cache
# --------------------------------------------------------------------------- #
def _cache_path(url: str) -> Path:
    return CACHE_DIR / f"{hashlib.sha1(url.encode()).hexdigest()}.json"


def _get_json(path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
    """GET a JSON endpoint, using the on-disk cache when possible.

    Returns {"_status": <code>, ...body} so callers can see 404s etc.
    """
    url = str(httpx.URL(BASE_URL + path, params=params or {}))
    cp = _cache_path(url)
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))

    try:
        r = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT,
                      follow_redirects=True)
    except httpx.HTTPError as e:  # network problem: don't cache
        return {"_status": 0, "error": f"network error: {e}"}

    try:
        body = r.json()
    except ValueError:
        body = {"error": "non-JSON response"}
    if not isinstance(body, dict):
        body = {"data": body}
    body["_status"] = r.status_code

    if r.status_code < 500:  # cache successes and 404s; retry server errors
        CACHE_DIR.mkdir(exist_ok=True)
        cp.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    return body


# --------------------------------------------------------------------------- #
# Plain functions (used by tests, app, and question builder)
# --------------------------------------------------------------------------- #
def search_brands(name: str, limit: int = 6) -> list[dict[str, Any]]:
    """Search the index and re-rank candidates by fuzzy name/domain similarity."""
    data = _get_json("/brands/search", {"q": name})
    items = data.get("items", [])
    q = name.lower().strip()
    for it in items:
        name_score = fuzz.WRatio(q, (it.get("name") or "").lower())
        domain_root = (it.get("domain") or "").lower().split(".")[0]
        domain_score = fuzz.ratio(q.replace(" ", ""), domain_root)
        it["match_score"] = round(max(name_score, domain_score), 1)
        it["page"] = f"{BASE_URL}/brand/{it['slug']}"
    slug_q = "-".join(q.split())
    # Tie-breakers: exact name, exact slug, then shorter (more canonical) names.
    items.sort(key=lambda x: (
        x["match_score"],
        (x.get("name") or "").lower() == q,
        x.get("slug") == slug_q,
        -len(x.get("name") or ""),
    ), reverse=True)
    return [
        {k: it.get(k) for k in ("slug", "name", "domain", "category", "match_score", "page")}
        for it in items[:limit]
    ]


def fetch_profile(slug: str) -> dict[str, Any]:
    """Return a compact, citation-friendly version of a brand profile."""
    data = _get_json(f"/brand/{slug}.json")
    if data.get("_status") != 200:
        return {"error": f"No profile for slug '{slug}' (HTTP {data.get('_status')})."}

    page = data.get("url") or f"{BASE_URL}/brand/{slug}"
    facts = []
    for header in data.get("headers", []):
        for s in header.get("sections", []):
            content = s.get("content")
            if isinstance(content, list):  # "links" shape, e.g. competitors
                content = ", ".join(c.get("name") or c.get("url") or "" for c in content)
            facts.append({
                "section": f"{header.get('header')} / {s.get('label')}",
                "content": content,
                "source": s.get("source_label"),
                "source_url": s.get("source_url"),
                "updated_at": (s.get("updated_at") or "")[:10],
            })
    return {
        "name": data.get("name"),
        "slug": slug,
        "domain": data.get("domain"),
        "categories": data.get("categories", []),
        "is_claimed": data.get("is_claimed"),
        "profile_url": page,  # canonical URL to cite
        "facts": facts,
    }


# --------------------------------------------------------------------------- #
# LangChain tools (what the agent sees)
# --------------------------------------------------------------------------- #
@tool
def search_brand(name: str) -> str:
    """Find brands in the Optimly Index whose name or domain matches `name`.

    Returns up to 6 candidates with slug, name, domain, category and a
    0-100 match_score. Several close matches mean the name may be ambiguous.
    """
    results = search_brands(name)
    if not results:
        return json.dumps({"candidates": [], "note": "No brands found."})
    return json.dumps({"candidates": results}, ensure_ascii=False)


@tool
def get_brand_profile(slug: str) -> str:
    """Get the public Optimly profile for one brand by its slug (from search_brand).

    Returns the brand's facts, each with the section it came from, its source,
    and the date. Cite `profile_url` as evidence.
    """
    return json.dumps(fetch_profile(slug), ensure_ascii=False)


TOOLS = [search_brand, get_brand_profile]
