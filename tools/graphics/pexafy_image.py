"""Stock image acquisition from Pexafy API (free plan, semantic search).

Pexafy aggregates free images from 9 sources (Pexels, Unsplash, Pixabay,
Kaboompics, Burst, ...) behind one API — the stand-in for `pexels_image`
now that Pexels API key signup is paused. Images only; video B-roll goes
through the stock video tools.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


class PexafyImage(BaseTool):
    name = "pexafy_image"
    version = "0.1.0"
    tier = ToolTier.SOURCE
    capability = "image_generation"
    provider = "pexafy"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set PEXAFY_API_KEY to your Pexafy API key.\n"
        "  Free plan (5K requests/month) at "
        "https://pexafy.com/dashboard/api-keys/create"
    )
    agent_skills = []

    capabilities = ["search_image", "download_image", "stock_image"]
    supports = {
        "orientation_filter": True,
        "color_filter": True,
        "semantic_search": True,
        "free_commercial_use": True,
    }
    best_for = [
        "real-world photography (cities, nature, people, objects)",
        "establishing shots and B-roll stills",
        "semantic scene descriptions — full sentences beat keywords",
        "free stock images when the Pexels key signup is unavailable",
    ]
    not_good_for = [
        "custom/specific compositions",
        "abstract or stylized graphics",
        "video clips (Pexafy is images-only)",
        "offline use",
    ]

    input_schema = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Plain-language scene description (max 500 chars). "
                    "Full descriptive sentences return the best matches."
                ),
            },
            "orientation": {
                "type": "string",
                "enum": ["landscape", "portrait", "square"],
            },
            "color": {
                "type": "string",
                "description": "Color name (e.g. red, blue, teal)",
            },
            "per_page": {"type": "integer", "default": 5, "minimum": 1, "maximum": 100},
            "score_threshold": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Minimum relevance to keep (0-1); omit for widest set",
            },
            "source": {
                "type": "string",
                "description": "Limit origin provider (e.g. Pexels, Unsplash, Pixabay)",
            },
            "download_size": {
                "type": "string",
                "enum": ["full", "large", "regular", "small", "thumb"],
                "default": "large",
                "description": "full=2400w, large=1920w, regular=1280w",
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["query", "orientation", "color", "source"]
    side_effects = ["writes image file to output_path", "calls Pexafy API"]
    user_visible_verification = ["Check that downloaded image matches the intended scene"]

    def get_status(self) -> ToolStatus:
        if os.environ.get("PEXAFY_API_KEY"):
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0  # Pexafy free plan

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = os.environ.get("PEXAFY_API_KEY")
        if not api_key:
            return ToolResult(
                success=False,
                error="PEXAFY_API_KEY not set. " + self.install_instructions,
            )

        import requests

        start = time.time()
        query = inputs["query"]

        params: dict[str, Any] = {
            "q": query[:500],
            "per_page": inputs.get("per_page", 5),
            "license_type": "free",
        }
        if inputs.get("orientation"):
            params["orientation"] = inputs["orientation"]
        if inputs.get("color"):
            params["color_name"] = inputs["color"]
        if inputs.get("source"):
            params["source"] = inputs["source"]
        if inputs.get("score_threshold") is not None:
            params["score_threshold"] = inputs["score_threshold"]

        try:
            search_response = requests.get(
                "https://api.pexafy.com/api/v1/search/photos",
                headers={"x-api-key": api_key},
                params=params,
                timeout=30,
            )
            search_response.raise_for_status()
            payload = search_response.json()

            if not payload.get("success", True):
                err = payload.get("error") or {}
                return ToolResult(
                    success=False,
                    error=f"Pexafy API error: {err.get('code', 'unknown')} {err.get('message', '')}".strip(),
                )

            photos = payload.get("data", [])
            if not photos:
                return ToolResult(
                    success=False,
                    error=f"No images found for query: {query}",
                    data={"results_returned": 0},
                )

            # Pick the first result (agent can refine query if needed)
            photo = photos[0]
            urls = photo.get("urls") or {}
            size = inputs.get("download_size", "large")
            image_url = urls.get(size) or urls.get("large") or photo.get("image_url", "")
            if not image_url:
                return ToolResult(success=False, error="Pexafy result has no usable URL")

            # Origin CDNs (Unsplash/Pixabay) TLS-flake intermittently —
            # retry the download itself, matching retry_policy.
            for attempt in range(3):
                try:
                    image_response = requests.get(image_url, timeout=60)
                    image_response.raise_for_status()
                    break
                except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                    if attempt == 2:
                        raise
                    time.sleep(1.5 * (attempt + 1))

            output_path = Path(
                inputs.get("output_path", f"pexafy_{photo.get('photo_id', 'image')}.jpg")
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(image_response.content)

        except Exception as e:
            return ToolResult(success=False, error=f"Pexafy image search failed: {e}")

        attribution = (photo.get("attribution") or {}).get("plain", "")
        return ToolResult(
            success=True,
            data={
                "provider": "pexafy",
                "photo_id": photo.get("photo_id"),
                "origin_source": photo.get("source", ""),
                "photographer": photo.get("photographer_full_name")
                or photo.get("photographer_username", "Unknown"),
                "photographer_url": photo.get("photographer_url", ""),
                "alt": photo.get("alt_description", ""),
                "width": photo.get("width"),
                "height": photo.get("height"),
                "relevance_score": photo.get("relevance_score"),
                "query": query,
                "output": str(output_path),
                "results_returned": len(photos),
                "license": f"Pexafy aggregated ({photo.get('source', 'unknown')} — {photo.get('license_type', 'free')})",
                "attribution": attribution,
                "pexafy_url": photo.get("source_image_url", ""),
            },
            artifacts=[str(output_path)],
            cost_usd=0.0,
            duration_seconds=round(time.time() - start, 2),
        )
