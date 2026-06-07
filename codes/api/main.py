from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from .routers import physics, optimization
from .security import APIKeyManager
import uvicorn

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if APIKeyManager.validate_key(api_key):
        return api_key
    raise HTTPException(
        status_code=403,
        detail="Could not validate API Key"
    )

app = FastAPI(
    title="DeVana REST API",
    description="High-performance API for Dynamic Vibration Absorber Optimization and Physics Simulation.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    dependencies=[Depends(get_api_key)]
)

# Include Routers
app.include_router(physics.router)
app.include_router(optimization.router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the DeVana REST API",
        "documentation": "/api/docs",
        "status": "online"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
