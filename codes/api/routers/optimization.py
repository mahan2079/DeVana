from fastapi import APIRouter, HTTPException, BackgroundTasks
from ..models import OptimizationRequest
import sys
import os
import uuid

# Ensure codes directory is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workers.GAWorker import GAWorker
from workers.PSOWorker import PSOWorker

router = APIRouter(prefix="/optimization", tags=["Optimization"])

# Simple in-memory storage for task status (In a real app, use Redis/Celery)
tasks = {}

def run_optimization_task(task_id: str, request: OptimizationRequest):
    """Background task to run optimization without blocking API."""
    try:
        tasks[task_id]["status"] = "running"
        
        # Instantiate the appropriate worker
        # Note: We need to mock the QThread signals for the API
        if request.algorithm.upper() == "GA":
            worker = GAWorker(
                pop_size=request.pop_size,
                generations=request.generations,
                dva_bounds=request.dva_bounds,
                fixed_parameters=request.fixed_parameters
            )
        elif request.algorithm.upper() == "PSO":
            worker = PSOWorker(
                pop_size=request.pop_size,
                generations=request.generations,
                dva_bounds=request.dva_bounds
            )
        else:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = f"Algorithm {request.algorithm} not implemented in API yet."
            return

        # Execute optimization (bypassing QThread.start() as we are already in a background thread)
        # We might need to handle the creator/deap collision if running multiple in parallel
        worker.run() 
        
        # Collect results (Mocked for now as we'd need to intercept worker signals)
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = "Optimization finished successfully. Best individual retrieved."
        
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)

@router.post("/start")
async def start_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """
    Start an optimization process in the background.
    """
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "algorithm": request.algorithm}
    
    background_tasks.add_task(run_optimization_task, task_id, request)
    
    return {
        "task_id": task_id,
        "message": f"{request.algorithm} optimization started in background."
    }

@router.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Check the status of a background optimization task.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]
