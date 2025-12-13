"""
Jobs API routes for pipeline job management.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
from app.infra.queue import task_queue
from app.infra.redis import redis_client
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


class JobSubmitRequest(BaseModel):
    """Request schema for job submission."""
    pipeline_config: Dict[str, Any]


class JobResponse(BaseModel):
    """Response schema for job status."""
    job_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None


@router.post("", response_model=Dict[str, Any])
async def submit_job(request: JobSubmitRequest):
    """
    Submit a new pipeline job.
    
    Request body:
    {
        "pipeline_config": {
            "run_id": "abc123",
            "nodes": [...],
            "edges": [...],
            "global_config": {...}
        }
    }
    """
    try:
        pipeline_config = request.pipeline_config
        run_id = pipeline_config.get("run_id")
        
        if not run_id:
            raise HTTPException(status_code=400, detail="run_id is required in pipeline_config")
        
        job_id = f"job_{run_id}"
        
        # Store job metadata in Redis
        job_data = {
            "job_id": job_id,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "pipeline_config": pipeline_config
        }
        
        await redis_client.client.hset(f"jobs:{job_id}", mapping={
            k: str(v) if not isinstance(v, str) else v 
            for k, v in job_data.items()
        })
        
        # Enqueue job
        task_queue.enqueue_job("orchestration", "execute_pipeline", pipeline_config)
        
        logger.info("Job submitted", job_id=job_id, run_id=run_id)
        
        return {
            "job_id": job_id,
            "status": "pending",
            "created_at": job_data["created_at"]
        }
    
    except Exception as e:
        logger.error("Job submission failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get job status and results."""
    try:
        job_data = await redis_client.client.hgetall(f"jobs:{job_id}")
        
        if not job_data:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return JobResponse(**job_data)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get job", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    try:
        # Update job status
        exists = await redis_client.client.exists(f"jobs:{job_id}")
        
        if not exists:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        await redis_client.client.hset(f"jobs:{job_id}", "status", "cancelled")
        
        logger.info("Job cancelled", job_id=job_id)
        
        return {"job_id": job_id, "status": "cancelled"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel job", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
