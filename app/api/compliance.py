from fastapi import APIRouter

from app.schemas.compliance import CacheMetricsResponse, PiiMetricsResponse
from app.services.pii_metrics_service import get_pii_metrics
from app.services.response_cache_service import get_cache_metrics

router = APIRouter(
    prefix="/compliance",
    tags=["Compliance"]
)


@router.get("/pii-metrics", response_model=PiiMetricsResponse)
async def pii_metrics():
    """
    Cumulative visibility into how much real traffic contains PII (by
    category), for a data-handling/DPA conversation with the AI
    provider - not a live operational signal, and not tied to whether
    a category is actually masked before the AI call (see
    pii_detector.py: this covers more categories than pii_masker.py
    masks).
    """

    return get_pii_metrics()


@router.get("/cache-metrics", response_model=CacheMetricsResponse)
async def cache_metrics():
    """
    Cumulative hit/miss counts for the exact-match response cache -
    grouped with the other read-only internal counters this router
    already exposes (pii-metrics), rather than a compliance concern
    itself. Not a live operational signal, just enough visibility to
    confirm the cache is actually reducing AI-provider calls on real
    traffic - see response_cache_service.py.
    """

    return get_cache_metrics()
