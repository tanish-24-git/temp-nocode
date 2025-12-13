"""
Export Agent for exporting trained models in various formats.
"""
import os
import tempfile
import shutil
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from app.agents.base_agent import BaseAgent
from app.export.adapter_export import export_adapter
from app.export.merged_export import export_merged
from app.export.gguf_export import export_gguf
from app.storage.object_store import object_store
from app.storage.model_registry import model_registry
from app.utils.exceptions import AgentException
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ExportInput(BaseModel):
    """Input schema for ExportAgent."""
    run_id: str
    model_path: str = Field(description="Path to model checkpoint in MinIO")
    base_model: str = Field(description="Base model name")
    export_formats: List[str] = Field(default=["adapter"], description="Export formats: adapter, merged, gguf")


class ExportAgent(BaseAgent):
    """
    Export agent for exporting models in various formats.
    """
    
    def __init__(self):
        super().__init__("ExportAgent")
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute model export.
        
        Args:
            input_data: {
                run_id: str,
                model_path: str,
                base_model: str,
                export_formats: List[str]
            }
        
        Returns:
            {
                exports: Dict[str, str]  # format -> MinIO path
            }
        """
        validated_input = self.validate_input(input_data, ExportInput)
        
        run_id = validated_input.run_id
        
        await self._emit_log(run_id, "INFO", "Starting model export")
        
        try:
            # Download model
            await self._emit_log(run_id, "INFO", "Downloading model")
            model_dir = await self._download_model(validated_input.model_path)
            
            # Load model
            await self._emit_log(run_id, "INFO", "Loading model")
            tokenizer = AutoTokenizer.from_pretrained(validated_input.base_model)
            base_model_obj = AutoModelForCausalLM.from_pretrained(validated_input.base_model)
            model = PeftModel.from_pretrained(base_model_obj, model_dir)
            
            exports = {}
            
            # Export in requested formats
            for format_name in validated_input.export_formats:
                await self._emit_log(run_id, "INFO", f"Exporting {format_name} format")
                
                export_dir = tempfile.mkdtemp(prefix=f"export_{format_name}_")
                
                if format_name == "adapter":
                    export_adapter(model, export_dir)
                elif format_name == "merged":
                    export_merged(model, tokenizer, export_dir)
                elif format_name == "gguf":
                    # First export merged, then convert to GGUF
                    merged_dir = tempfile.mkdtemp(prefix="merged_")
                    export_merged(model, tokenizer, merged_dir)
                    export_gguf(merged_dir, export_dir)
                    shutil.rmtree(merged_dir, ignore_errors=True)
                else:
                    await self._emit_log(run_id, "WARN", f"Unknown export format: {format_name}")
                    continue
                
                # Upload to MinIO
                minio_path = await self._upload_export(run_id, export_dir, format_name)
                exports[format_name] = minio_path
                
                # Update model registry
                model_registry.add_export(run_id, format_name, minio_path)
                
                # Cleanup
                shutil.rmtree(export_dir, ignore_errors=True)
            
            await self._emit_log(run_id, "INFO", "Export completed", exports=list(exports.keys()))
            
            # Cleanup
            shutil.rmtree(model_dir, ignore_errors=True)
            
            return {"exports": exports}
        
        except Exception as e:
            await self._emit_log(run_id, "ERROR", f"Export failed: {str(e)}")
            raise AgentException("EXPORT_ERROR", str(e))
    
    async def _download_model(self, model_path: str) -> str:
        """Download model from MinIO."""
        local_dir = tempfile.mkdtemp(prefix="model_")
        objects = object_store.list_objects(prefix=model_path, bucket_type='checkpoints')
        
        for obj_name in objects:
            local_file = os.path.join(local_dir, os.path.basename(obj_name))
            object_store.download_file(obj_name, local_file, bucket_type='checkpoints')
        
        return local_dir
    
    async def _upload_export(self, run_id: str, export_dir: str, format_name: str) -> str:
        """Upload export to MinIO."""
        export_path = f"{run_id}/exports/{format_name}"
        
        for root, dirs, files in os.walk(export_dir):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, export_dir)
                object_name = f"{export_path}/{relative_path}"
                
                object_store.upload_file(local_file, object_name, bucket_type='artifacts')
        
        return export_path
