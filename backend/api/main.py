from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.user_routes import router as user_router
from backend.api.image_routes import router as image_router

app = FastAPI(title="AI Matching System", description="User matching and image recommendation system", version="1.0.0")

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# Include routers
app.include_router(user_router, prefix="/api/users", tags=["users"])
app.include_router(image_router, prefix="/api/images", tags=["images"])


@app.get("/")
async def root():
    """
    Root endpoint returning API information
    """
    return {"name": "AI Matching System API", "version": "1.0.0", "status": "running"}
