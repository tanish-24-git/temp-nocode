"""
Projects API routes for multi-project management.
Each project can have multiple fine-tuning jobs and datasets.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from app.infra.redis import redis_client
from app.utils.logging import get_logger
import uuid

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


# Pydantic Models

class ProjectCreate(BaseModel):
    """Request schema for creating a project."""
    project_name: str = Field(..., min_length=1, max_length=100, description="Project name")
    description: Optional[str] = Field(None, max_length=500, description="Project description")
    tags: List[str] = Field(default_factory=list, description="Tags for organization")


class ProjectResponse(BaseModel):
    """Response schema for project."""
    project_id: str
    project_name: str
    description: Optional[str]
    tags: List[str]
    created_at: str
    updated_at: str
    job_count: int = 0
    dataset_count: int = 0


class ProjectUpdate(BaseModel):
    """Request schema for updating a project."""
    project_name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    tags: Optional[List[str]] = None


# Endpoints

@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(project: ProjectCreate):
    """
    Create a new project.
    
    Projects organize datasets and training jobs.
    Example: "CustomerSupportBot", "FinancialAnalyzer"
    """
    try:
        project_id = f"proj_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow().isoformat()
        
        project_data = {
            "project_id": project_id,
            "project_name": project.project_name,
            "description": project.description or "",
            "tags": ",".join(project.tags),
            "created_at": now,
            "updated_at": now,
            "job_count": "0",
            "dataset_count": "0"
        }
        
        # Store in Redis
        await redis_client.client.hset(f"projects:{project_id}", mapping=project_data)
        
        # Add to projects index
        await redis_client.client.sadd("projects:index", project_id)
        
        logger.info("Project created", project_id=project_id, name=project.project_name)
        
        return ProjectResponse(
            project_id=project_id,
            project_name=project.project_name,
            description=project.description,
            tags=project.tags,
            created_at=now,
            updated_at=now,
            job_count=0,
            dataset_count=0
        )
    
    except Exception as e:
        logger.error("Failed to create project", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")


@router.get("", response_model=List[ProjectResponse])
async def list_projects(
    limit: int = Query(50, ge=1, le=100, description="Max projects to return"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """
    List all projects with pagination.
    """
    try:
        # Get all project IDs
        project_ids = await redis_client.client.smembers("projects:index")
        
        if not project_ids:
            return []
        
        # Apply pagination
        project_ids = list(project_ids)[offset:offset + limit]
        
        projects = []
        for project_id in project_ids:
            data = await redis_client.client.hgetall(f"projects:{project_id}")
            
            if data:
                projects.append(ProjectResponse(
                    project_id=data["project_id"],
                    project_name=data["project_name"],
                    description=data.get("description") or None,
                    tags=data.get("tags", "").split(",") if data.get("tags") else [],
                    created_at=data["created_at"],
                    updated_at=data["updated_at"],
                    job_count=int(data.get("job_count", 0)),
                    dataset_count=int(data.get("dataset_count", 0))
                ))
        
        logger.info("Listed projects", count=len(projects))
        return projects
    
    except Exception as e:
        logger.error("Failed to list projects", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {str(e)}")


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str):
    """Get project details by ID."""
    try:
        data = await redis_client.client.hgetall(f"projects:{project_id}")
        
        if not data:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
        
        return ProjectResponse(
            project_id=data["project_id"],
            project_name=data["project_name"],
            description=data.get("description") or None,
            tags=data.get("tags", "").split(",") if data.get("tags") else [],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            job_count=int(data.get("job_count", 0)),
            dataset_count=int(data.get("dataset_count", 0))
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get project", project_id=project_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get project: {str(e)}")


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: str, updates: ProjectUpdate):
    """Update project details."""
    try:
        # Check if exists
        exists = await redis_client.client.exists(f"projects:{project_id}")
        if not exists:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
        
        # Prepare updates
        update_data = {}
        if updates.project_name is not None:
            update_data["project_name"] = updates.project_name
        if updates.description is not None:
            update_data["description"] = updates.description
        if updates.tags is not None:
            update_data["tags"] = ",".join(updates.tags)
        
        update_data["updated_at"] = datetime.utcnow().isoformat()
        
        # Apply updates
        if update_data:
            await redis_client.client.hset(f"projects:{project_id}", mapping=update_data)
        
        # Get updated data
        data = await redis_client.client.hgetall(f"projects:{project_id}")
        
        logger.info("Project updated", project_id=project_id)
        
        return ProjectResponse(
            project_id=data["project_id"],
            project_name=data["project_name"],
            description=data.get("description") or None,
            tags=data.get("tags", "").split(",") if data.get("tags") else [],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            job_count=int(data.get("job_count", 0)),
            dataset_count=int(data.get("dataset_count", 0))
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update project", project_id=project_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update project: {str(e)}")


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    """
    Delete a project.
    
    WARNING: This will not delete associated jobs/datasets, only the project metadata.
    """
    try:
        exists = await redis_client.client.exists(f"projects:{project_id}")
        if not exists:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
        
        # Delete project data
        await redis_client.client.delete(f"projects:{project_id}")
        
        # Remove from index
        await redis_client.client.srem("projects:index", project_id)
        
        logger.info("Project deleted", project_id=project_id)
        
        return {"project_id": project_id, "status": "deleted"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete project", project_id=project_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")
