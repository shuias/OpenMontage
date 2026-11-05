"""Pexafy stock image source adapter.

Wraps the Pexafy semantic image-search API behind the unified
`StockSource` protocol. Pexafy aggregates free-to-use images from 9
providers (Pexels, Unsplash, Pixabay, Kaboompics, Burst, ...) into one
API — added as the Pexels key signup was paused, so it stands in for
the image side of `PexelsSource`.

IMPORTANT: Pexafy is images-only. It has no video endpoint, so
`kind="video"` searches return an empty list; video B-roll still comes
from the other adapters (videvo, mixkit, coverr, pixabay_video, ...).

Search is natural-language: full descriptive sentences outperform bare
keywords. Pagination is cursor-based, so `SearchFilters.page` is not
honoured — page 1 semantics apply to every call.

Uses `PEXAFY_API_KEY` from the environment. `.env` is loaded at process
start by `tools.base_tool._load_dotenv`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .base import Candidate, SearchFilters


_SEARCH_URL = "https://api.pexafy.com/api/v1/search/photos"


class PexafySource:
    """Pexafy adapter for stock images.

    Satisfies `StockSource`. Stateless — all config comes from the
    environment, so the corpus builder can instantiate one per run
    without caching the instance.
    """

    name = "pexafy"
    display_name = "Pexafy"
    provider = "pexafy"
    priority = 20
    install_instructions = (
        "Set PEXAFY_API_KEY in .env to enable Pexafy image search "
        "(free plan, 5K requests/month, key at "
        "https://pexafy.com/dashboard/api-keys/create)."
    )
    supports = {"video": False, "image": True}

    def is_available(self) -> bool:
        return bool(os.environ.get("PEXAFY_API_KEY"))

    # ------------------------------------------------------------------
    # Public protocol
    # ------------------------------------------------------------------

    def search(self, query: str, filters: SearchFilters) -> list[Candidate]:
        """Search Pexafy images. `kind="video"` returns nothing.

        The corpus builder's `kind="any"` fan-out still works — we just
        contribute the image half of the results.
        """
        kind = (filters.kind or "video").lower()
        if kind == "video":
            return []
        return self._search_images(query, filters)

    def download(self, candidate: Candidate, out_path: Path) -> Path:
        """Stream the candidate's file to `out_path`.

        Same streamed-chunk semantics as the other adapters; caching is
        the corpus builder's responsibility.
        """
        import requests  # lazy — avoid pulling requests at import time

        if not candidate.download_url:
            raise ValueError(
                f"Candidate {candidate.clip_id} has no download_url"
            )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(
            candidate.download_url, stream=True, timeout=120
        ) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
        return out_path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        key = os.environ.get("PEXAFY_API_KEY")
        if not key:
            raise RuntimeError(
                "PEXAFY_API_KEY not set. Get a free key at "
                "https://pexafy.com/dashboard/api-keys/create and add it to .env."
            )
        return {"x-api-key": key}

    def _search_images(
        self, query: str, filters: SearchFilters
    ) -> list[Candidate]:
        import requests  # lazy

        params: dict[str, Any] = {
            "q": query[:500],
            "per_page": max(1, min(filters.per_page, 100)),
            "license_type": "free",
        }
        if filters.orientation:
            params["orientation"] = filters.orientation

        r = requests.get(
            _SEARCH_URL,
            headers=self._headers(),
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if not data.get("success", True):
            err = data.get("error") or {}
            raise RuntimeError(f"Pexafy search error: {err}")
        photos = data.get("data", []) or []

        out: list[Candidate] = []
        for p in photos:
            width = int(p.get("width", 0) or 0)
            height = int(p.get("height", 0) or 0)
            if filters.min_width is not None and width < filters.min_width:
                continue

            urls = p.get("urls") or {}
            # "large" is the w=1920 rendition — the sweet spot for
            # montage use. Fall back through the size ladder.
            download_url = (
                urls.get("large")
                or urls.get("full")
                or p.get("image_url")
                or ""
            )
            if not download_url:
                continue

            license_type = (p.get("license_type") or "free").strip()
            attribution = (p.get("attribution") or {}).get("plain", "")
            tags = (p.get("alt_description") or p.get("description") or "").strip()

            out.append(
                Candidate(
                    source=self.name,
                    source_id=str(p.get("photo_id", "")),
                    source_url=p.get("source_image_url", "") or "",
                    download_url=download_url,
                    kind="image",
                    width=width,
                    height=height,
                    duration=0.0,
                    creator=(
                        p.get("photographer_full_name")
                        or p.get("photographer_username")
                        or ""
                    ),
                    license=(
                        f"Pexafy aggregated ({p.get('source', 'unknown')} — "
                        f"{license_type})"
                    ),
                    source_tags=tags,
                    thumbnail_url=urls.get("small") or urls.get("thumb") or "",
                    extra={
                        "origin_source": p.get("source"),
                        "relevance_score": p.get("relevance_score"),
                        "color_hex": p.get("color_hex"),
                        "orientation": p.get("orientation"),
                        "attribution_plain": attribution,
                    },
                )
            )
        return out
