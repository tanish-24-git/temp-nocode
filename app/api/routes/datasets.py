"""
Datasets API routes for dataset management.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
from app.storage.object_store import object_store
from app.utils.logging import get_logger
import uuid
import tempfile
import os

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file."""
    try:
        dataset_id = f"ds_{uuid.uuid4().hex[:12]}"
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Upload to MinIO
        object_name = f"{dataset_id}/{file.filename}"
        object_store.upload_file(tmp_path, object_name, bucket_type='datasets')
        
        # Cleanup
        os.unlink(tmp_path)
        
        logger.info("Dataset uploaded", dataset_id=dataset_id, filename=file.filename)
        
        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "size_bytes": len(content),
            "s3_path": f"s3://datasets/{object_name}"
        }
    
    except Exception as e:
        logger.error("Dataset upload failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get dataset information."""
    try:
        # List objects in dataset
        objects = object_store.list_objects(prefix=dataset_id, bucket_type='datasets')
        
        if not objects:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        return {
            "dataset_id": dataset_id,
            "files": objects,
            "s3_path": f"s3://datasets/{dataset_id}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get dataset", dataset_id=dataset_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, limit: int = 10):
    """Preview dataset samples."""
    # Placeholder - would need to download and parse dataset
    return {
        "dataset_id": dataset_id,
        "samples": [],
        "note": "Preview not yet implemented"
    }
