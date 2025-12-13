"""
Merged model export utilities.
Merges LoRA adapter into base model and exports full model.
"""
import os
from app.utils.logging import get_logger

logger = get_logger(__name__)


def export_merged(model, tokenizer, output_dir: str) -> str:
    """
    Export merged model (adapter + base model).
    
    Args:
        model: PEFT model with LoRA
        tokenizer: Tokenizer
        output_dir: Directory to save merged model
    
    Returns:
        Path to exported model
    """
    logger.info("Exporting merged model", output_dir=output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge adapter with base model
    merged_model = model.merge_and_unload()
    
    # Save merged model
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Merged model exported successfully", path=output_dir)
    
    return output_dir
