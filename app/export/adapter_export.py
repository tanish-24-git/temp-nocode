"""
LoRA adapter export utilities.
Exports only the adapter weights (small file size).
"""
import os
from pathlib import Path
from app.utils.logging import get_logger

logger = get_logger(__name__)


def export_adapter(model, output_dir: str) -> str:
    """
    Export LoRA adapter weights.
    
    Args:
        model: PEFT model with LoRA
        output_dir: Directory to save adapter
    
    Returns:
        Path to exported adapter
    """
    logger.info("Exporting LoRA adapter", output_dir=output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save adapter weights
    model.save_pretrained(output_dir)
    
    logger.info("LoRA adapter exported successfully", path=output_dir)
    
    return output_dir
