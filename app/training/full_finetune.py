"""
Full fine-tuning implementation.
Trains all model parameters (not parameter-efficient).
Requires significantly more memory than LoRA/QLoRA.
"""
import torch
from typing import Dict, Any, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from app.utils.logging import get_logger

logger = get_logger(__name__)


class FullFinetuneTrainer:
    """
    Full fine-tuning trainer with CPU/GPU support.
    Trains all model parameters - requires more memory than LoRA/QLoRA.
    """
    
    def __init__(
        self,
        base_model: str,
        device: str = "auto"
    ):
        """
        Initialize full fine-tuning trainer.
        
        Args:
            base_model: HuggingFace model name or path
            device: Device to use ('cpu', 'cuda:0', or 'auto')
        """
        self.base_model = base_model
        self.device = self._get_device(device)
        
        logger.info(
            "Full fine-tune trainer initialized",
            base_model=base_model,
            device=self.device
        )
    
    def _get_device(self, device: str) -> str:
        """Get appropriate device (CPU/GPU)."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model_and_tokenizer(self):
        """
        Load base model and tokenizer for full fine-tuning.
        All parameters will be trainable.
        """
        logger.info("Loading base model and tokenizer", model=self.base_model)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate dtype for device
        if self.device == "cpu":
            # Use float32 for CPU
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        else:
            # Use float16 for GPU to save memory
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.device_count() > 1 else None
            )
            if torch.cuda.device_count() == 1:
                model = model.to(self.device)
        
        # All parameters are trainable in full fine-tuning
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(
            "Model loaded for full fine-tuning",
            total_params=total_params,
            trainable_params=trainable_params,
            device=self.device
        )
        
        return model, tokenizer
    
    def train(
        self,
        model,
        tokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "./output",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,  # Lower LR for full fine-tuning
        max_seq_length: int = 2048,
        gradient_accumulation_steps: int = 4,
        save_steps: int = 100,
        logging_steps: int = 10,
        warmup_steps: int = 100,
        callbacks: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Train model with full fine-tuning.
        
        Args:
            model: Base model
            tokenizer: Tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            output_dir: Output directory for checkpoints
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate (typically lower than LoRA)
            max_seq_length: Maximum sequence length
            gradient_accumulation_steps: Gradient accumulation steps
            save_steps: Save checkpoint every N steps
            logging_steps: Log metrics every N steps
            warmup_steps: Number of warmup steps
            callbacks: Training callbacks
        
        Returns:
            Training metrics dictionary
        """
        logger.info(
            "Starting full fine-tuning",
            epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=self.device
        )
        
        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=self.device != "cpu",  # Use FP16 only on GPU
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if eval_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=self.device != "cpu",
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks or []
        )
        
        # Train
        train_result = trainer.train()
        
        # Save final model
        trainer.save_model(output_dir)
        
        logger.info(
            "Full fine-tuning completed",
            train_loss=train_result.training_loss,
            epochs=num_epochs
        )
        
        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime"),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
            "total_steps": train_result.global_step
        }
