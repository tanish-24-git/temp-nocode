"""
Intelligent hyperparameter suggestion service.
Analyzes dataset characteristics and suggests optimal training configuration.
"""
from typing import Dict, Any, Optional
from app.ai.ollama_client import ollama_client
from app.infra.gpu_manager import gpu_manager
from app.utils.logging import get_logger
import json

logger = get_logger(__name__)


class ParamSuggester:
    """
    Intelligent hyperparameter suggestion based on dataset characteristics.
    
    Uses TinyLlama to provide context-aware recommendations with reasoning.
    """
    
    def __init__(self):
        self.gpu_available = gpu_manager.has_gpu()
    
    async def suggest_config(
        self,
        dataset_stats: Dict[str, Any],
        task_type: Optional[str] = None,
        gpu_count: int = 0
    ) -> Dict[str, Any]:
        """
        Suggest training configuration based on dataset and hardware.
        
        Args:
            dataset_stats: {
                "rows": int,
                "avg_text_length": float,
                "num_classes": int (for classification),
                "label_distribution": dict (optional)
            }
            task_type: "classification", "chat", "summarization", etc.
            gpu_count: Number of available GPUs
        
        Returns:
            {
                "batch_size": int,
                "epochs":int,
                "lora_rank": int,
                "lora_alpha": int,
                "learning_rate": float,
                "precision": "fp16" | "fp32",
                "reasoning": str  # AI explanation
            }
        """
        logger.info("Suggesting hyperparameters", dataset_rows=dataset_stats.get("rows"), gpu_count=gpu_count)
        
        # Extract stats
        num_rows = dataset_stats.get("rows", 0)
        avg_length = dataset_stats.get("avg_text_length", 100)
        num_classes = dataset_stats.get("num_classes", 2)
        
        # Rule-based baseline (fast, deterministic)
        baseline_config = self._rule_based_suggestion(
            num_rows, avg_length, num_classes, gpu_count
        )
        
        # AI enhancement (adds reasoning and fine-tuning)
        try:
            ai_config = await self._ai_enhanced_suggestion(
                dataset_stats, task_type, baseline_config, gpu_count
            )
            return ai_config
        except Exception as e:
            logger.warning("AI suggestion failed, using baseline", error=str(e))
            baseline_config["reasoning"] = "Using rule-based defaults (AI unavailable)"
            return baseline_config
    
    def _rule_based_suggestion(
        self,
        num_rows: int,
        avg_length: float,
        num_classes: int,
        gpu_count: int
    ) -> Dict[str, Any]:
        """
        Rule-based hyperparameter suggestion (deterministic fallback).
        
        Rules:
        - Small dataset (<1k rows): Low rank, more epochs
        - Medium dataset (1k-10k): Balanced
        - Large dataset (>10k): Higher rank, fewer epochs
        - GPU: fp16, larger batch
        - CPU: fp32, smaller batch
        """
        # Defaults
        config = {
            "batch_size": 4,
            "epochs": 3,
            "lora_rank": 16,
            "lora_alpha": 32,
            "learning_rate": 2e-4,
            "precision": "fp16" if gpu_count > 0 else "fp32"
        }
        
        # Adjust for dataset size
        if num_rows < 1000:
            # Small dataset: Lower rank to prevent overfitting
            config["lora_rank"] = 8
            config["lora_alpha"] = 16
            config["epochs"] = 5
            config["learning_rate"] = 1e-4
        elif num_rows < 10000:
            # Medium dataset: Balanced
            config["lora_rank"] = 16
            config["lora_alpha"] = 32
            config["epochs"] = 3
        else:
            # Large dataset: Higher rank
            config["lora_rank"] = 32
            config["lora_alpha"] = 64
            config["epochs"] = 2
        
        # Adjust batch size for GPU
        if gpu_count > 0:
            config["batch_size"] = 8 if num_rows > 5000 else 4
        else:
            config["batch_size"] = 2  # CPU: smaller batch
        
        # Adjust for text length
        if avg_length > 500:
            # Long texts: smaller batch, lower rank
            config["batch_size"] = max(2, config["batch_size"] // 2)
            config["lora_rank"] = max(8, config["lora_rank"] // 2)
        
        return config
    
    async def _ai_enhanced_suggestion(
        self,
        dataset_stats: Dict[str, Any],
        task_type: Optional[str],
        baseline_config: Dict[str, Any],
        gpu_count: int
    ) -> Dict[str, Any]:
        """
        Use TinyLlama to enhance baseline suggestion with reasoning.
        """
        # Construct prompt for AI
        system_prompt = """You are an expert in LLM fine-tuning. Analyze the dataset and suggest hyperparameters.
Respond ONLY with valid JSON, no other text.
Format: {
  "batch_size": <int>,
  "epochs": <int>,
  "lora_rank": <int>,
  "lora_alpha": <int>,
  "learning_rate": <float>,
  "reasoning": "<one sentence explanation>"
}"""
        
        user_prompt = f"""Dataset:
- Rows: {dataset_stats.get('rows')}
- Avg text length: {dataset_stats.get('avg_text_length', 'unknown')}
- Task type: {task_type or 'general'}
- Classes: {dataset_stats.get('num_classes', 'N/A')}
- GPU count: {gpu_count}

Baseline suggestion:
{json.dumps(baseline_config, indent=2)}

Provide optimal hyperparameters as JSON."""
        
        # Call TinyLlama
        response = await ollama_client.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.3,  # Low temp for more consistent output
            max_tokens=300
        )
        
        # Parse JSON response
        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()
            
            ai_config = json.loads(response)
            
            # Ensure precision is included
            ai_config["precision"] = baseline_config["precision"]
            
            logger.info("AI suggestion success", ai_config=ai_config)
            return ai_config
        
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse AI response as JSON", error=str(e), response=response[:200])
            # Extract reasoning from non-JSON response and keep baseline
            baseline_config["reasoning"] = f"AI analysis: {response[:200]}"
            return baseline_config


# Global instance
param_suggester = ParamSuggester()
