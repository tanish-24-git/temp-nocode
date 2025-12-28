"""
Training Agent for LLM fine-tuning.
Supports LoRA, QLoRA, and full fine-tuning methods with automatic CPU/GPU detection.
"""
import os
import uuid
import tempfile
from typing import Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
from datasets import load_from_disk, Dataset
from app.agents.base_agent import BaseAgent
from app.training.lora import LoRATrainer
from app.training.qlora import QLoRATrainer
from app.training.full_finetune import FullFinetuneTrainer
from app.infra.gpu_manager import gpu_manager
from app.infra.logging_stream import LogStream
from app.storage.object_store import object_store
from app.storage.model_registry import model_registry
from app.utils.exceptions import AgentException
from app.utils.logging import get_logger

logger = get_logger(__name__)


class TrainingInput(BaseModel):
    """Input schema for TrainingAgent."""
    run_id: str
    dataset_path: str = Field(description="Path to preprocessed dataset in MinIO")
    base_model: str = Field(description="HuggingFace model name or path")
    training_config: Dict[str, Any] = Field(description="Training configuration")


class TrainingOutput(BaseModel):
    """Output schema for TrainingAgent."""
    model_id: str
    checkpoint_path: str
    final_metrics: Dict[str, Any]


class TrainingAgent(BaseAgent):
    """
    Training agent for LLM fine-tuning.
    
    Supports:
    - LoRA: Parameter-efficient fine-tuning
    - QLoRA: 4-bit quantized LoRA (GPU only, falls back to LoRA on CPU)
    - Full: Full model fine-tuning
    
    Automatically detects GPU availability and adjusts settings accordingly.
    """
    
    def __init__(self):
        super().__init__("TrainingAgent")
        self.gpu_available = gpu_manager.has_gpu()
        self.device = gpu_manager.get_device() if self.gpu_available else "cpu"
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute training job.
        
        Args:
            input_data: {
                run_id: str,
                dataset_path: str,  # MinIO path
                base_model: str,
                training_config: {
                    method: "lora" | "qlora" | "full",
                    lora_r: int (default: 16),
                    lora_alpha: int (default: 32),
                    lora_dropout: float (default: 0.05),
                    epochs: int (default: 3),
                    batch_size: int (default: 4),
                    learning_rate: float (default: 2e-4),
                    max_seq_length: int (default: 2048),
                    gradient_accumulation_steps: int (default: 4),
                    save_steps: int (default: 100)
                }
            }
        
        Returns:
            {
                model_id: str,
                checkpoint_path: str,
                final_metrics: dict
            }
        """
        # Validate input
        validated_input = self.validate_input(input_data, TrainingInput)
        
        run_id = validated_input.run_id
        dataset_path = validated_input.dataset_path
        base_model = validated_input.base_model
        config = validated_input.training_config
        
        # Extract training parameters
        method = config.get("method", "lora").lower()
        lora_r = config.get("lora_r", 16)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.05)
        epochs = config.get("epochs", 3)
        batch_size = config.get("batch_size", 4)
        learning_rate = config.get("learning_rate", 2e-4)
        max_seq_length = config.get("max_seq_length", 2048)
        gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4)
        save_steps = config.get("save_steps", 100)
        
        await self._emit_log(
            run_id,
            "INFO",
            f"Starting training with {method.upper()} method",
            base_model=base_model,
            device=self.device,
            gpu_available=self.gpu_available,
            method=method
        )
        
        try:
            # Download dataset from MinIO
            await self._emit_log(run_id, "INFO", "Downloading dataset from MinIO")
            local_dataset_dir = await self._download_dataset(dataset_path)
            
            # Load dataset
            await self._emit_log(run_id, "INFO", "Loading dataset")
            train_dataset = load_from_disk(local_dataset_dir)
            
            await self._emit_log(
                run_id,
                "INFO",
                f"Dataset loaded: {len(train_dataset)} samples",
                num_samples=len(train_dataset)
            )
            
            # Initialize trainer based on method
            await self._emit_log(run_id, "INFO", f"Initializing {method.upper()} trainer")
            
            if method == "lora":
                trainer = LoRATrainer(
                    base_model=base_model,
                    device=self.device,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout
                )
            elif method == "qlora":
                trainer = QLoRATrainer(
                    base_model=base_model,
                    device=self.device,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout
                )
            elif method == "full":
                trainer = FullFinetuneTrainer(
                    base_model=base_model,
                    device=self.device
                )
            else:
                raise AgentException(
                    "TRAINING_ERROR",
                    f"Unknown training method: {method}. Use 'lora', 'qlora', or 'full'."
                )
            
            # Load model and tokenizer
            await self._emit_log(run_id, "INFO", "Loading model and tokenizer")
            model, tokenizer = trainer.load_model_and_tokenizer()
            
            # Create output directory
            output_dir = tempfile.mkdtemp(prefix=f"training_{run_id}_")
            
            # Train model
            await self._emit_log(
                run_id,
                "INFO",
                f"Starting training: {epochs} epochs, batch size {batch_size}",
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            # Note: Callbacks would be added here for real-time metric streaming
            # For now, we'll train without callbacks (they require async support in Trainer)
            metrics = trainer.train(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                output_dir=output_dir,
                num_epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_seq_length=max_seq_length,
                gradient_accumulation_steps=gradient_accumulation_steps,
                save_steps=save_steps,
                logging_steps=10
            )
            
            await self._emit_log(
                run_id,
                "INFO",
                "Training completed",
                final_loss=metrics.get("train_loss"),
                total_steps=metrics.get("total_steps")
            )
            
            # Upload checkpoint to MinIO
            await self._emit_log(run_id, "INFO", "Uploading checkpoint to MinIO")
            checkpoint_path = await self._upload_checkpoint(run_id, output_dir)
            
            # Generate model ID
            model_id = f"model_{run_id}_{uuid.uuid4().hex[:8]}"
            
            # Register model in registry
            await self._emit_log(run_id, "INFO", "Registering model")
            model_registry.register_model(
                run_id=run_id,
                model_name=f"{base_model.split('/')[-1]}-finetuned",
                base_model=base_model,
                dataset_id=dataset_path,
                training_config=config,
                metrics=metrics,
                model_path=checkpoint_path
            )
            
            # Clean up local files
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
            shutil.rmtree(local_dataset_dir, ignore_errors=True)
            
            return {
                "model_id": model_id,
                "checkpoint_path": checkpoint_path,
                "final_metrics": metrics
            }
        
        except Exception as e:
            await self._emit_log(
                run_id,
                "ERROR",
                f"Training failed: {str(e)}",
                error=str(e)
            )
            raise AgentException("TRAINING_ERROR", str(e))
    
    async def _download_dataset(self, dataset_path: str) -> str:
        """Download dataset from MinIO to local temp directory."""
        local_dir = tempfile.mkdtemp(prefix="dataset_")
        
        # List all files in dataset path
        objects = object_store.list_objects(prefix=dataset_path, bucket_type='datasets')
        
        for obj_name in objects:
            # Download each file
            local_file = os.path.join(local_dir, os.path.basename(obj_name))
            object_store.download_file(obj_name, local_file, bucket_type='datasets')
        
        return local_dir
    
    async def _upload_checkpoint(self, run_id: str, checkpoint_dir: str) -> str:
        """Upload checkpoint directory to MinIO."""
        checkpoint_path = f"{run_id}/checkpoint"
        
        # Upload all files in checkpoint directory
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, checkpoint_dir)
                object_name = f"{checkpoint_path}/{relative_path}"
                
                object_store.upload_file(
                    local_file,
                    object_name,
                    bucket_type='checkpoints'
                )
        
        return checkpoint_path
