"""
AI Assistant API routes.
Provides endpoints for AI-powered fine-tuning assistance.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from app.ai.ollama_client import ollama_client
from app.ai.param_suggest import param_suggester
from app.ai.task_detector import task_detector
from app.ai.assistant import ai_assistant
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/ai", tags=["ai"])


# Request/Response Models

class DetectTaskRequest(BaseModel):
    """Request for task detection."""
    samples: List[Dict[str, Any]] = Field(description="5-10 sample rows from dataset")
    column_names: List[str] = Field(description="List of column names")


class DetectTaskResponse(BaseModel):
    """Response for task detection."""
    task_type: str
    confidence: float
    target_column: str
    reasoning: str


class SuggestConfigRequest(BaseModel):
    """Request for config suggestion."""
    dataset_stats: Dict[str, Any] = Field(description="Dataset statistics")
    task_type: Optional[str] = Field(None, description="Optional task type")
    gpu_count: int = Field(0, description="Number of available GPUs")


class SuggestConfigResponse(BaseModel):
    """Response for config suggestion."""
    batch_size: int
    epochs: int
    lora_rank: int
    lora_alpha: int
    learning_rate: float
    precision: str
    reasoning: str


class ExplainMetricsRequest(BaseModel):
    """Request for metrics explanation."""
    metrics: Dict[str, Any]
    training_logs: Optional[List[Dict[str, Any]]] = None


class ExplainMetricsResponse(BaseModel):
    """Response for metrics explanation."""
    explanation: str


class DiagnoseErrorsRequest(BaseModel):
    """Request for error diagnosis."""
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    context: Optional[Dict[str, Any]] = None


class DiagnoseErrorsResponse(BaseModel):
    """Response for error diagnosis."""
    diagnosis: str
    suggested_fixes: List[str]
    severity: str


class ChatRequest(BaseModel):
    """Request for AI chat."""
    message: str
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Response for AI chat."""
    response: str


# Endpoints

@router.get("/health")
async def ai_health():
    """Check if AI service (Ollama) is healthy."""
    try:
        is_healthy = await ollama_client.health_check()
        
        if is_healthy:
            return {
                "status": "healthy",
                "model": ollama_client.model_name,
                "service": "ollama"
            }
        else:
            raise HTTPException(status_code=503, detail="Ollama service unhealthy")
    
    except Exception as e:
        logger.error("AI health check failed", error=str(e))
        raise HTTPException(status_code=503, detail=f"AI service unavailable: {str(e)}")


@router.post("/detect-task", response_model=DetectTaskResponse)
async def detect_task(request: DetectTaskRequest):
    """
    Auto-detect task type from dataset samples.
    
    Analyzes sample rows and column names to determine:
    - classification
    - qa (question-answering)
    - chat
    - summarization
    - generation
    - extraction
    """
    try:
        logger.info("Task detection request", num_samples=len(request.samples))
        
        result = await task_detector.detect_task(
            samples=request.samples,
            column_names=request.column_names
        )
        
        logger.info("Task detection complete", task_type=result["task_type"])
        return DetectTaskResponse(**result)
    
    except Exception as e:
        logger.error("Task detection failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Task detection failed: {str(e)}")


@router.post("/suggest-config", response_model=SuggestConfigResponse)
async def suggest_config(request: SuggestConfigRequest):
    """
    Suggest optimal training configuration based on dataset characteristics.
    
    Analyzes:
    - Dataset size (rows)
    - Text length
    - Task type
    - GPU availability
    
    Returns intelligent suggestions for:
    - Batch size
    - Epochs
    - LoRA rank/alpha
    - Learning rate
    - Precision (fp16/fp32)
    """
    try:
        logger.info("Config suggestion request", dataset_rows=request.dataset_stats.get("rows"))
        
        result = await param_suggester.suggest_config(
            dataset_stats=request.dataset_stats,
            task_type=request.task_type,
            gpu_count=request.gpu_count
        )
        
        logger.info("Config suggestion complete", batch_size=result["batch_size"])
        return SuggestConfigResponse(**result)
    
    except Exception as e:
        logger.error("Config suggestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Config suggestion failed: {str(e)}")


@router.post("/explain-metrics", response_model=ExplainMetricsResponse)
async def explain_metrics(request: ExplainMetricsRequest):
    """
    Explain training metrics in human-readable format.
    
    Provides insights into:
    - What the metrics mean
    - Whether training was successful
    - Suggestions for improvement
    """
    try:
        logger.info("Metrics explanation request", num_metrics=len(request.metrics))
        
        explanation = await ai_assistant.explain_metrics(
            metrics=request.metrics,
            training_logs=request.training_logs
        )
        
        logger.info("Metrics explanation complete")
        return ExplainMetricsResponse(explanation=explanation)
    
    except Exception as e:
        logger.error("Metrics explanation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Metrics explanation failed: {str(e)}")


@router.post("/diagnose-errors", response_model=DiagnoseErrorsResponse)
async def diagnose_errors(request: DiagnoseErrorsRequest):
    """
    Diagnose errors and warnings, suggest fixes.
    
    Analyzes:
    - Error messages
    - Warning messages
    - Context (dataset, config)
    
    Returns:
    - Root cause diagnosis
    - Suggested fixes
    - Severity assessment
    """
    try:
        logger.info("Error diagnosis request", num_errors=len(request.errors), num_warnings=len(request.warnings))
        
        diagnosis = await ai_assistant.diagnose_errors(
            errors=request.errors,
            warnings=request.warnings,
            context=request.context
        )
        
        logger.info("Error diagnosis complete", severity=diagnosis["severity"])
        return DiagnoseErrorsResponse(**diagnosis)
    
    except Exception as e:
        logger.error("Error diagnosis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error diagnosis failed: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    General chat with AI assistant about fine-tuning.
    
    Ask questions about:
    - Training methods (LoRA, QLoRA, Full)
    - Hyperparameter tuning
    - Debugging issues
    - Best practices
    """
    try:
        logger.info("AI chat request", message_length=len(request.message))
        
        response = await ai_assistant.chat(
            message=request.message,
            context=request.context
        )
        
        logger.info("AI chat complete", response_length=len(response))
        return ChatResponse(response=response)
    
    except Exception as e:
        logger.error("AI chat failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"AI chat failed: {str(e)}")
