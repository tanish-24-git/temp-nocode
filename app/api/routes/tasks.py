"""
Tasks API routes for task type detection and configuration suggestions.
Analyzes datasets to auto-detect ML task type and suggest optimal settings.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from app.ai.task_detector import task_detector
from app.ai.param_suggest import param_suggester
from app.utils.logging import get_logger
import pandas as pd
import io

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


# Pydantic Models

class TaskSuggestionRequest(BaseModel):
    """Request for task type suggestion from dataset samples."""
    samples: List[Dict[str, Any]] = Field(..., max_items=20, description="5-20 sample rows")
    column_names: List[str] = Field(..., min_items=1, description="Dataset column names")
    domain: Optional[Literal["general", "finance", "medical", "legal", "code", "custom"]] = "general"
    language: Optional[Literal["en", "es", "fr", "de", "multi"]] = "en"


class TaskSuggestionResponse(BaseModel):
    """Response with suggested task type and configuration."""
    task_type: Literal["classification", "regression", "chat", "summarization", "qa", "extraction"]
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    target_column: str
    reasoning: str
    suggested_config: Dict[str, Any]


class TrainingPresetRequest(BaseModel):
    """Request for training configuration preset."""
    training_mode: Literal["fast", "balanced", "high_quality"]
    dataset_stats: Dict[str, Any]
    gpu_count: int = Field(0, ge=0, le=8)
    task_type: Optional[str] = None


class TrainingPresetResponse(BaseModel):
    """Response with training configuration preset."""
    preset_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    lora_rank: int
    lora_alpha: int
    precision: str
    gradient_accumulation: int
    reasoning: str


# Endpoints

@router.post("/suggest", response_model=TaskSuggestionResponse)
async def suggest_task(request: TaskSuggestionRequest):
    """
    Auto-detect task type from dataset samples.
    
    Analyzes column names and data patterns to determine:
    - classification (sentiment, category)
    - regression (price prediction)
    - chat (instruction-following)
    - summarization (text -> summary)
    - qa (question -> answer)
    - extraction (entity extraction)
    
    Example request:
    ```json
    {
      "samples": [
        {"text": "Great product!", "label": "positive"},
        {"text": "Terrible service", "label": "negative"}
      ],
      "column_names": ["text", "label"],
      "domain": "general"
    }
    ```
    """
    try:
        logger.info("Task suggestion request", num_samples=len(request.samples), domain=request.domain)
        
        # Detect task using AI
        detection_result = await task_detector.detect_task(
            samples=request.samples,
            column_names=request.column_names
        )
        
        # Get optimal config for detected task
        dataset_stats = {
            "rows": len(request.samples) * 100,  # Estimate from samples
            "avg_text_length": sum(len(str(v)) for row in request.samples for v in row.values()) / max(len(request.samples), 1),
            "num_classes": len(set(row.get(detection_result["target_column"], "") for row in request.samples)) if detection_result["task_type"] == "classification" else None
        }
        
        suggested_config_result = await param_suggester.suggest_config(
            dataset_stats=dataset_stats,
            task_type=detection_result["task_type"],
            gpu_count=0  # Will be provided in actual training request
        )
        
        logger.info(
            "Task suggestion complete",
            task_type=detection_result["task_type"],
            confidence=detection_result["confidence"]
        )
        
        return TaskSuggestionResponse(
            task_type=detection_result["task_type"],
            confidence=detection_result["confidence"],
            target_column=detection_result["target_column"],
            reasoning=detection_result["reasoning"],
            suggested_config={
                "batch_size": suggested_config_result["batch_size"],
                "epochs": suggested_config_result["epochs"],
                "lora_rank": suggested_config_result["lora_rank"],
                "learning_rate": suggested_config_result["learning_rate"],
                "precision": suggested_config_result["precision"]
            }
        )
    
    except Exception as e:
        logger.error("Task suggestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Task suggestion failed: {str(e)}")


@router.post("/preset", response_model=TrainingPresetResponse)
async def get_training_preset(request: TrainingPresetRequest):
    """
    Get training configuration preset based on mode.
    
    Presets:
    - **fast**: Quick training for testing (1-2 epochs, small batch)
    - **balanced**: Good balance of speed and quality (3 epochs)
    - **high_quality**: Best results, slower (5+ epochs, optimal params)
    
    Example:
    ```json
    {
      "training_mode": "balanced",
      "dataset_stats": {"rows": 10000, "avg_text_length": 150},
      "gpu_count": 1,
      "task_type": "classification"
    }
    ```
    """
    try:
        logger.info("Training preset request", mode=request.training_mode, gpu_count=request.gpu_count)
        
        # Get AI-suggested config
        config = await param_suggester.suggest_config(
            dataset_stats=request.dataset_stats,
            task_type=request.task_type,
            gpu_count=request.gpu_count
        )
        
        # Apply preset adjustments
        if request.training_mode == "fast":
            config["epochs"] = 1
            config["batch_size"] = max(config["batch_size"], 4)  # Larger batch for speed
            config["lora_rank"] = min(8, config["lora_rank"])  # Smaller rank
            reasoning = "Fast mode: 1 epoch, larger batch size for quick results"
        
        elif request.training_mode == "balanced":
            config["epochs"] = 3
            reasoning = "Balanced mode: 3 epochs with optimal settings"
        
        else:  # high_quality
            config["epochs"] = 5
            config["learning_rate"] = config["learning_rate"] * 0.5  # Lower LR for stability
            config["lora_rank"] = min(32, config["lora_rank"] * 2)  # Higher rank
            reasoning = "High quality mode: 5 epochs, lower LR, higher rank for best results"
        
        # Calculate gradient accumulation
        gradient_accumulation = 1
        if request.gpu_count == 0:  # CPU
            gradient_accumulation = 4  # Compensate for small batch
        elif config["batch_size"] < 8:
            gradient_accumulation = 8 // config["batch_size"]
        
        logger.info("Training preset generated", mode=request.training_mode, epochs=config["epochs"])
        
        return TrainingPresetResponse(
            preset_name=request.training_mode,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            lora_rank=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            precision=config["precision"],
            gradient_accumulation=gradient_accumulation,
            reasoning=reasoning + f". {config.get('reasoning', '')}"
        )
    
    except Exception as e:
        logger.error("Preset generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Preset generation failed: {str(e)}")


@router.post("/analyze-csv")
async def analyze_csv_for_task(file: UploadFile = File(...)):
    """
    Upload a CSV file and get task type + config suggestions.
    
    This is a convenience endpoint that combines file upload with task detection.
    Returns suggested task type and optimal training configuration.
    """
    try:
        logger.info("CSV analysis request", filename=file.filename)
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Get samples (up to 20 rows)
        samples = df.head(20).to_dict('records')
        column_names = df.columns.tolist()
        
        # Detect task
        detection_result = await task_detector.detect_task(
            samples=samples,
            column_names=column_names
        )
        
        # Get config
        dataset_stats = {
            "rows": len(df),
            "avg_text_length": df.astype(str).apply(lambda x: x.str.len()).mean().mean(),
            "num_classes": df[detection_result["target_column"]].nunique() if detection_result["task_type"] == "classification" else None
        }
        
        config_result = await param_suggester.suggest_config(
            dataset_stats=dataset_stats,
            task_type=detection_result["task_type"],
            gpu_count=0
        )
        
        logger.info(
            "CSV analysis complete",
            task_type=detection_result["task_type"],
            rows=len(df)
        )
        
        return {
            "task_detection": detection_result,
            "suggested_config": config_result,
            "dataset_stats": dataset_stats
        }
    
    except Exception as e:
        logger.error("CSV analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"CSV analysis failed: {str(e)}")
