# services/preprocessing_service.py
from pathlib import Path
from typing import Dict, Any, List, Optional
import anyio
import json
import structlog

from config.settings import settings
from services.preprocessing_engine import PreprocessingEngine

logger = structlog.get_logger()

_engine = PreprocessingEngine(settings.upload_directory)

def preprocess_dataset(file_path: str, plan: Dict[str, Any]) -> str:
    """
    Thin synchronous wrapper expected by main.py and Background tasks.
    Accepts file_path (string) and plan dict (as described in preprocessing_plan.md).
    Returns preprocessed file path string.
    """
    # Validate basic shape
    if not isinstance(plan, dict):
        raise ValueError("plan must be a dict")

    # Run in threadpool to avoid blocking event loop (callers already use anyio.to_thread.run_sync)
    # The engine itself uses pandas; callers already call via to_thread in main.py
    res = _engine.run_tabular(file_path, plan, provenance=plan.get("provenance"))
    return res["preprocessed_path"]
