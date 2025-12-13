"""
Models API routes for trained model management.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.storage.model_registry import model_registry
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("")
async def list_models(
    limit: int = Query(20, le=100),
    offset: int = Query(0, ge=0),
    base_model: Optional[str] = None
):
    """List all trained models with pagination."""
    try:
        models = model_registry.list_models(prefix="")
        
        # Apply base_model filter if specified
        if base_model:
            models = [m for m in models if m.get("base_model") == base_model]
        
        # Pagination
        total = len(models)
        paginated_models = models[offset:offset + limit]
        
        return {
            "models": paginated_models,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }
    
    except Exception as e:
        logger.error("Failed to list models", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}")
async def get_model(run_id: str):
    """Get model details by run ID."""
    try:
        model_metadata = model_registry.get_model_metadata(run_id)
        
        if not model_metadata:
            raise HTTPException(status_code=404, detail=f"Model for run {run_id} not found")
        
        return model_metadata
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model", run_id=run_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/card")
async def get_model_card(run_id: str):
    """Get model card in Markdown format."""
    try:
        model_metadata = model_registry.get_model_metadata(run_id)
        
        if not model_metadata:
            raise HTTPException(status_code=404, detail=f"Model for run {run_id} not found")
        
        card = model_registry.generate_model_card(run_id, model_metadata)
        
        return {"content": card, "format": "markdown"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model card", run_id=run_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/download")
async def download_model(run_id: str, format: str = "adapter"):
    """Get download URL for model."""
    try:
        model_metadata = model_registry.get_model_metadata(run_id)
        
        if not model_metadata:
            raise HTTPException(status_code=404, detail=f"Model for run {run_id} not found")
        
        exports = model_metadata.get("exports", {})
        
        if format not in exports:
            raise HTTPException(
                status_code=404,
                detail=f"Export format '{format}' not available. Available: {list(exports.keys())}"
            )
        
        export_path = exports[format]["path"]
        
        return {
            "format": format,
            "path": export_path,
            "note": "Download from MinIO using this path"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to download model", run_id=run_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
