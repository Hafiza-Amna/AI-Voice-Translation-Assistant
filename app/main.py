from fastapi import FastAPI
from app.config import settings
from app.api.v1 import api_router

app = FastAPI(
    title=settings.APP_NAME,
    description="FastAPI Backend for AI Voice Translation Assistant (Urdu <-> English)",
    version="1.0.0-mvp",
    debug=settings.DEBUG,
)

# Include V1 API Routers
app.include_router(api_router)

@app.get("/")
async def root():
    """Root endpoint verifying API availability."""
    return {
        "message": f"Welcome to {settings.APP_NAME} API",
        "docs_url": "/docs",
        "health_check": "/api/v1/health"
    }
