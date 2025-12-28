"""
Export Agent for converting and packaging models.
Handles export to .safetensors, ONNX, and GGUF formats.
"""
import os
import shutil
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from app.agents.base_agent import BaseAgent
from app.storage.object_store import object_store
from app.utils.logging import get_logger

logger = get_logger(__name__)

class ExportInput(BaseModel):
    """Input for ExportAgent."""
    run_id: str
    model_id: str
    formats: List[str] = ["safetensors", "gguf"]

class ExportOutput(BaseModel):
    """Output for ExportAgent."""
    artifacts: Dict[str, str]  # format -> s3_path

class ExportAgent(BaseAgent):
    """
    Exports trained models to various formats.
    """
    
    def __init__(self):
        super().__init__("ExportAgent")
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run export pipeline.
        
        1. Download model checkpoint
        2. Convert to requested formats
        3. Upload artifacts to MinIO
        """
        validated_input = self.validate_input(input_data, ExportInput)
        run_id = validated_input.run_id
        formats = validated_input.formats
        
        await self._emit_log(run_id, "INFO", f"Starting model export: {formats}")
        
        # Mock export logic
        # Real logic: use PEFT merge_and_unload(), llama.cpp, etc.
        
        artifacts = {}
        
        for fmt in formats:
            await self._emit_log(run_id, "INFO", f"Exporting to {fmt}...")
            
            # Simulate export
            s3_path = f"models/{run_id}/{fmt}/adapter.{fmt if fmt != 'gguf' else 'bin'}"
            artifacts[fmt] = s3_path
            
            await self._emit_log(run_id, "INFO", f"Exported {fmt} to {s3_path}")
            
        return {
            "artifacts": artifacts,
            "model_card": f"models/{run_id}/README.md"
        }
