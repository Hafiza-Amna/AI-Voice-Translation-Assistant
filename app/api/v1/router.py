from fastapi import APIRouter
from app.config import settings
from app.api.v1.translation import router as translation_router

router = APIRouter()
router.include_router(translation_router, tags=["Audio Translation"])

@router.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint returning application status."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "version": "1.0.0-mvp"
    }
