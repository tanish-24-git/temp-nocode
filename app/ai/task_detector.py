"""
Task type detection service.
Automatically detects the ML task type from dataset samples using TinyLlama.
"""
from typing import Dict, Any, List
from app.ai.ollama_client import ollama_client
from app.utils.logging import get_logger
import json

logger = get_logger(__name__)


class TaskDetector:
    """
    Automatically detect task type from dataset samples.
    
    Uses TinyLlama to analyze row structure and infer:
    - classification
    - chat
    - summarization
    - question-answering
    - text-generation
    - extraction
    """
    
    async def detect_task(
        self,
        samples: List[Dict[str, Any]],
        column_names: List[str]
    ) -> Dict[str, Any]:
        """
        Detect task type from dataset samples.
        
        Args:
            samples: List of 5-10 sample rows from dataset
            column_names: List of column names
        
        Returns:
            {
                "task_type": str,  # classification, chat, summarization, qa, generation, extraction
                "confidence": float,  # 0-1
                "target_column": str,  # Suggested output column
                "reasoning": str
            }
        """
        logger.info("Detecting task type", num_samples=len(samples), columns=column_names)
        
        # Quick rule-based detection first
        rule_based_result = self._rule_based_detection(samples, column_names)
        
        # AI enhancement
        try:
            ai_result = await self._ai_detection(samples, column_names, rule_based_result)
            return ai_result
        except Exception as e:
            logger.warning("AI detection failed, using rule-based", error=str(e))
            return rule_based_result
    
    def _rule_based_detection(
        self,
        samples: List[Dict[str, Any]],
        column_names: List[str]
    ) -> Dict[str, Any]:
        """
        Rule-based task detection (fast fallback).
        
        Rules:
        - Has 'label' or 'class' column → classification
        - Has 'question' + 'answer' → qa
        - Has 'instruction' + 'response' → chat
        - Has 'text' + 'summary' → summarization
        - Otherwise → generation
        """
        col_lower = [c.lower() for c in column_names]
        
        # Classification indicators
        if 'label' in col_lower or 'class' in col_lower or 'category' in col_lower:
            return {
                "task_type": "classification",
                "confidence": 0.7,
                "target_column": next(c for c in column_names if c.lower() in ['label', 'class', 'category']),
                "reasoning": "Dataset has label/class column"
            }
        
        # QA indicators
        if ('question' in col_lower and 'answer' in col_lower):
            return {
                "task_type": "qa",
                "confidence": 0.8,
                "target_column": next(c for c in column_names if 'answer' in c.lower()),
                "reasoning": "Dataset has question-answer structure"
            }
        
        # Chat indicators
        if (('instruction' in col_lower or 'prompt' in col_lower) and 
            ('response' in col_lower or 'output' in col_lower)):
            return {
                "task_type": "chat",
                "confidence": 0.8,
                "target_column": next(c for c in column_names if c.lower() in ['response', 'output']),
                "reasoning": "Dataset has instruction-response format"
            }
        
        # Summarization indicators
        if 'summary' in col_lower:
            return {
                "task_type": "summarization",
                "confidence": 0.75,
                "target_column": next(c for c in column_names if 'summary' in c.lower()),
                "reasoning": "Dataset has summary column"
            }
        
        # Default to generation
        return {
            "task_type": "generation",
            "confidence": 0.5,
            "target_column": column_names[-1] if column_names else "text",
            "reasoning": "No clear task indicators, assuming text generation"
        }
    
    async def _ai_detection(
        self,
        samples: List[Dict[str, Any]],
        column_names: List[str],
        baseline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use TinyLlama to detect task type from samples."""
        
        # Format samples for AI
        sample_text = "\n".join([
            f"Row {i+1}: {json.dumps(row, indent=None)[:200]}"
            for i, row in enumerate(samples[:5])
        ])
        
        system_prompt = """You are an ML task classifier. Analyze the dataset samples and determine the task type.
Respond ONLY with valid JSON:
{
  "task_type": "classification|qa|chat|summarization|generation|extraction",
  "confidence": 0.0-1.0,
  "target_column": "column_name",
  "reasoning": "one sentence"
}"""
        
        user_prompt = f"""Columns: {', '.join(column_names)}

Sample rows:
{sample_text}

Baseline detection: {baseline['task_type']} (confidence: {baseline['confidence']})

Provide your task detection as JSON."""
        
        response = await ollama_client.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.2,
            max_tokens=200
        )
        
        # Parse JSON
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response)
            logger.info("AI task detection success", result=result)
            return result
        
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse AI detection response", error=str(e))
            return baseline


# Global instance
task_detector = TaskDetector()
