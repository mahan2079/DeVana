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
    
    print("\n" + "="*60)
    print("🚀 DeVana API is running!")
    print("="*60)
    print("📡 Port: 8000")
    print("🔑 API Keys loaded:")
    for user, key in keys.items():
        print(f"   - {user}: {key}")
    
    sample_key = list(keys.values())[0] if keys else "YOUR_API_KEY"
    
    print("\n🔗 How to connect from other software:")
    print("Include the following header in your HTTP requests:")
    print("X-API-Key: <your_api_key>")
    print("\nExample using curl:")
    print(f"curl -X GET http://localhost:8000/ -H \"X-API-Key: {sample_key}\"")
    print("\nExample using Python (requests):")
    print("import requests")
    print(f"headers = {{'X-API-Key': '{sample_key}'}}")
    print("response = requests.get('http://localhost:8000/', headers=headers)")
    print("print(response.json())")
    print("="*60 + "\n")
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
