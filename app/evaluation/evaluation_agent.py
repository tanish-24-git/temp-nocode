"""
Evaluation Agent for model evaluation.
Computes metrics on test datasets.
"""
import os
import tempfile
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from app.agents.base_agent import BaseAgent
from app.evaluation.metrics import compute_all_metrics
from app.infra.gpu_manager import gpu_manager
from app.storage.object_store import object_store
from app.utils.exceptions import AgentException
from app.utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationInput(BaseModel):
    """Input schema for EvaluationAgent."""
    run_id: str
    model_path: str = Field(description="Path to model checkpoint in MinIO")
    base_model: str = Field(description="Base model name")
    test_dataset_path: str = Field(description="Path to test dataset in MinIO")
    metrics: List[str] = Field(default=["bleu", "rouge", "accuracy"], description="Metrics to compute")
    is_adapter: bool = Field(default=True, description="Whether model is LoRA adapter")


class EvaluationAgent(BaseAgent):
    """
    Evaluation agent for computing model metrics.
    Supports CPU and GPU evaluation.
    """
    
    def __init__(self):
        super().__init__("EvaluationAgent")
        self.device = gpu_manager.get_device() if gpu_manager.has_gpu() else "cpu"
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute evaluation.
        
        Args:
            input_data: {
                run_id: str,
                model_path: str,
                base_model: str,
                test_dataset_path: str,
                metrics: List[str],
                is_adapter: bool
            }
        
        Returns:
            {
                metrics: Dict[str, float]
            }
        """
        validated_input = self.validate_input(input_data, EvaluationInput)
        
        run_id = validated_input.run_id
        
        await self._emit_log(run_id, "INFO", "Starting model evaluation", device=self.device)
        
        try:
            # Download model and dataset
            await self._emit_log(run_id, "INFO", "Downloading model and dataset")
            model_dir = await self._download_model(validated_input.model_path)
            dataset_dir = await self._download_dataset(validated_input.test_dataset_path)
            
            # Load model
            await self._emit_log(run_id, "INFO", "Loading model")
            model, tokenizer = await self._load_model(
                validated_input.base_model,
                model_dir if validated_input.is_adapter else None
            )
            
            # Load test dataset
            test_dataset = load_from_disk(dataset_dir)
            
            # Generate predictions
            await self._emit_log(run_id, "INFO", f"Generating predictions for {len(test_dataset)} samples")
            predictions, references = await self._generate_predictions(
                model, tokenizer, test_dataset
            )
            
            # Compute metrics
            await self._emit_log(run_id, "INFO", "Computing metrics")
            metrics = compute_all_metrics(
                model, tokenizer, predictions, references,
                device=self.device,
                compute_ppl="perplexity" in validated_input.metrics
            )
            
            await self._emit_log(run_id, "INFO", "Evaluation completed", metrics=metrics)
            
            # Cleanup
            import shutil
            shutil.rmtree(model_dir, ignore_errors=True)
            shutil.rmtree(dataset_dir, ignore_errors=True)
            
            return {"metrics": metrics}
        
        except Exception as e:
            await self._emit_log(run_id, "ERROR", f"Evaluation failed: {str(e)}")
            raise AgentException("EVALUATION_ERROR", str(e))
    
    async def _download_model(self, model_path: str) -> str:
        """Download model from MinIO."""
        local_dir = tempfile.mkdtemp(prefix="model_")
        objects = object_store.list_objects(prefix=model_path, bucket_type='checkpoints')
        
        for obj_name in objects:
            local_file = os.path.join(local_dir, os.path.basename(obj_name))
            object_store.download_file(obj_name, local_file, bucket_type='checkpoints')
        
        return local_dir
    
    async def _download_dataset(self, dataset_path: str) -> str:
        """Download dataset from MinIO."""
        local_dir = tempfile.mkdtemp(prefix="dataset_")
        objects = object_store.list_objects(prefix=dataset_path, bucket_type='datasets')
        
        for obj_name in objects:
            local_file = os.path.join(local_dir, os.path.basename(obj_name))
            object_store.download_file(obj_name, local_file, bucket_type='datasets')
        
        return local_dir
    
    async def _load_model(self, base_model: str, adapter_path: str = None):
        """Load model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        if self.device == "cpu":
            model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="float32")
        else:
            model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="float16").to(self.device)
        
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        
        return model, tokenizer
    
    async def _generate_predictions(self, model, tokenizer, dataset) -> tuple:
        """Generate predictions for dataset."""
        predictions = []
        references = []
        
        model.eval()
        
        for sample in dataset:
            input_text = sample.get("input", sample.get("text", ""))
            reference = sample.get("output", sample.get("label", ""))
            
            inputs = tokenizer(input_text, return_tensors="pt").to(self.device)
            outputs = model.generate(**inputs, max_new_tokens=100)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            predictions.append(prediction)
            references.append(reference)
        
        return predictions, references
