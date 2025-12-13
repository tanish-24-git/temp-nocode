"""
LoRA (Low-Rank Adaptation) training implementation.
Supports both CPU and GPU training with automatic device detection.
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
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LoRATrainer:
    """
    LoRA fine-tuning trainer with CPU/GPU support.
    Uses PEFT library for parameter-efficient fine-tuning.
    """
    
    def __init__(
        self,
        base_model: str,
        device: str = "auto",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None
    ):
        """
        Initialize LoRA trainer.
        
        Args:
            base_model: HuggingFace model name or path
            device: Device to use ('cpu', 'cuda:0', or 'auto')
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability
            target_modules: Modules to apply LoRA (None = auto-detect)
        """
        self.base_model = base_model
        self.device = self._get_device(device)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        
        logger.info(
            "LoRA trainer initialized",
            base_model=base_model,
            device=self.device,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )
    
    def _get_device(self, device: str) -> str:
        """Get appropriate device (CPU/GPU)."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model_and_tokenizer(self):
        """
        Load base model and tokenizer with LoRA configuration.
        Automatically handles CPU/GPU placement.
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
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        logger.info(
            "Model loaded with LoRA",
            trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
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
        learning_rate: float = 2e-4,
        max_seq_length: int = 2048,
        gradient_accumulation_steps: int = 4,
        save_steps: int = 100,
        logging_steps: int = 10,
        callbacks: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Train model with LoRA.
        
        Args:
            model: PEFT model with LoRA
            tokenizer: Tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            output_dir: Output directory for checkpoints
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            max_seq_length: Maximum sequence length
            gradient_accumulation_steps: Gradient accumulation steps
            save_steps: Save checkpoint every N steps
            logging_steps: Log metrics every N steps
            callbacks: Training callbacks
        
        Returns:
            Training metrics dictionary
        """
        logger.info(
            "Starting LoRA training",
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
            report_to="none",  # Disable wandb/tensorboard
            remove_unused_columns=False,
            dataloader_pin_memory=self.device != "cpu",  # Pin memory only on GPU
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
            "LoRA training completed",
            train_loss=train_result.training_loss,
            epochs=num_epochs
        )
        
        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime"),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
            "total_steps": train_result.global_step
        }
