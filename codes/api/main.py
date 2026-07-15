from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from contextlib import asynccontextmanager
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check if api_keys.json exists or has keys
    keys = APIKeyManager.load_keys()
    if not keys:
        default_key = APIKeyManager.generate_key()
        APIKeyManager.save_key("default_user", default_key)
        keys = {"default_user": default_key}
    
    print("\n" + "="*60, flush=True)
    print("🚀 DeVana API is running!", flush=True)
    print("="*60, flush=True)
    print("📡 Port: 8000", flush=True)
    print("🔑 API Keys loaded:", flush=True)
    for user, key in keys.items():
        print(f"   - {user}: {key}", flush=True)
    
    sample_key = list(keys.values())[0] if keys else "YOUR_API_KEY"
    
    print("\n🔗 How to connect from other software:", flush=True)
    print("Include the following header in your HTTP requests:", flush=True)
    print("X-API-Key: <your_api_key>", flush=True)
    print("\nExample using curl:", flush=True)
    print(f"curl -X GET http://localhost:8000/ -H \"X-API-Key: {sample_key}\"", flush=True)
    print("\nExample using Python (requests):", flush=True)
    print("import requests", flush=True)
    print(f"headers = {{'X-API-Key': '{sample_key}'}}", flush=True)
    print("response = requests.get('http://localhost:8000/', headers=headers)", flush=True)
    print("print(response.json())", flush=True)
    print("="*60 + "\n", flush=True)
    yield

app = FastAPI(
    title="DeVana REST API",
    description="High-performance API for Dynamic Vibration Absorber Optimization and Physics Simulation.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    dependencies=[Depends(get_api_key)],
    lifespan=lifespan
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
