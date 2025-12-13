"""
QLoRA (Quantized LoRA) training implementation.
Uses 4-bit quantization for memory-efficient training on GPU.
Falls back to regular LoRA on CPU.
"""
import torch
from typing import Dict, Any, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset
from app.utils.logging import get_logger

logger = get_logger(__name__)


class QLoRATrainer:
    """
    QLoRA fine-tuning trainer with 4-bit quantization.
    Automatically falls back to regular LoRA on CPU (quantization requires GPU).
    """
    
    def __init__(
        self,
        base_model: str,
        device: str = "auto",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None,
        use_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_quant_type: str = "nf4"
    ):
        """
        Initialize QLoRA trainer.
        
        Args:
            base_model: HuggingFace model name or path
            device: Device to use ('cpu', 'cuda:0', or 'auto')
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability
            target_modules: Modules to apply LoRA (None = auto-detect)
            use_4bit: Use 4-bit quantization (GPU only)
            bnb_4bit_compute_dtype: Compute dtype for 4-bit
            bnb_4bit_quant_type: Quantization type ('nf4' or 'fp4')
        """
        self.base_model = base_model
        self.device = self._get_device(device)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.use_4bit = use_4bit and self.device != "cpu"  # Quantization only on GPU
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        
        if self.device == "cpu" and use_4bit:
            logger.warning(
                "4-bit quantization not supported on CPU, falling back to regular LoRA"
            )
        
        logger.info(
            "QLoRA trainer initialized",
            base_model=base_model,
            device=self.device,
            use_4bit=self.use_4bit,
            lora_r=lora_r
        )
    
    def _get_device(self, device: str) -> str:
        """Get appropriate device (CPU/GPU)."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model_and_tokenizer(self):
        """
        Load base model and tokenizer with QLoRA configuration.
        Uses 4-bit quantization on GPU, regular precision on CPU.
        """
        logger.info("Loading base model and tokenizer", model=self.base_model)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure quantization if using GPU
        if self.use_4bit:
            logger.info("Using 4-bit quantization (QLoRA)")
            
            compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,  # Double quantization for extra memory savings
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map="auto" if torch.cuda.device_count() > 1 else None,
                torch_dtype=compute_dtype
            )
            
            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(model)
        else:
            # CPU or no quantization
            logger.info("Using regular precision (no quantization)")
            
            if self.device == "cpu":
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            else:
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
            "Model loaded with QLoRA",
            trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
            device=self.device,
            quantized=self.use_4bit
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
        Train model with QLoRA.
        
        Args:
            model: PEFT model with QLoRA
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
            "Starting QLoRA training",
            epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=self.device,
            quantized=self.use_4bit
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
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=self.device != "cpu",
            optim="paged_adamw_8bit" if self.use_4bit else "adamw_torch",  # Use 8-bit optimizer for QLoRA
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
            "QLoRA training completed",
            train_loss=train_result.training_loss,
            epochs=num_epochs
        )
        
        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime"),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
            "total_steps": train_result.global_step
        }
