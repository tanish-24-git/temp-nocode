"""
AI Assistant service for chat, metrics explanation, and error diagnosis.
"""
from typing import Dict, Any, List, Optional
from app.ai.ollama_client import ollama_client
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AIAssistant:
    """
    General purpose AI assistant for fine-tuning guidance.
    
    Capabilities:
    - Chat with users about fine-tuning
    - Explain training metrics and curves
    - Diagnose errors and provide fixes
    - General Q&A about LLMs
    """
    
    async def chat(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        General chat with AI assistant.
        
        Args:
            message: User message/question
            context: Optional context (dataset stats, training config, etc)
        
        Returns:
            AI response
        """
        logger.info("AI chat", message_length=len(message), has_context=context is not None)
        
        # Build system prompt with context
        system_prompt = """You are an expert AI assistant for LLM fine-tuning.
You help users:
- Choose the right training method (LoRA, QLoRA, Full)  
- Understand training metrics
- Optimize hyperparameters
- Debug training issues

Be concise, practical, and provide actionable advice."""
        
        # Add context if available
        if context:
            context_str = f"\n\nCurrent context:\n{self._format_context(context)}"
            system_prompt += context_str
        
        # Call TinyLlama
        response = await ollama_client.generate(
            prompt=message,
            system=system_prompt,
            temperature=0.7,
            max_tokens=400
        )
        
        logger.info("AI chat response", response_length=len(response))
        return response
    
    async def explain_metrics(
        self,
        metrics: Dict[str, Any],
        training_logs: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Explain training metrics in human-readable format.
        
        Args:
            metrics: Final metrics dict (loss, accuracy, etc)
            training_logs: Optional list of training step logs
        
        Returns:
            Human-readable explanation
        """
        logger.info("Explaining metrics", metrics_count=len(metrics))
        
        # Format metrics for AI
        metrics_text = self._format_metrics(metrics, training_logs)
        
        system_prompt = """You are an expert at explaining machine learning metrics.
Analyze the training metrics and provide:
1. What they mean
2. Whether training is successful
3. Suggestions for improvement

Be concise (2-3 sentences)."""
        
        prompt = f"""Training results:
{metrics_text}

Explain these metrics and assess training quality."""
        
        response = await ollama_client.generate(
            prompt=prompt,
            system=system_prompt,
            temperature=0.5,
            max_tokens=300
        )
        
        return response
    
    async def diagnose_errors(
        self,
        errors: List[str],
        warnings: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Diagnose errors and suggest fixes.
        
        Args:
            errors: List of error messages
            warnings: List of warning messages
            context: Optional context (dataset info, config)
        
        Returns:
            {
                "diagnosis": str,
                "suggested_fixes": List[str],
                "severity": "low|medium|high"
            }
        """
        logger.info("Diagnosing errors", num_errors=len(errors), num_warnings=len(warnings))
        
        # Format issues
        issues_text = "Errors:\n" + "\n".join(f"- {e}" for e in errors) if errors else ""
        issues_text += "\n\nWarnings:\n" + "\n".join(f"- {w}" for w in warnings) if warnings else ""
        
        system_prompt = """You are an expert at debugging LLM fine-tuning issues.
Analyze errors/warnings and provide:
1. Root cause diagnosis
2. Concrete suggested fixes
3. Severity assessment

Format response as:
DIAGNOSIS: <one sentence>
FIXES:
- <fix 1>
- <fix 2>
SEVERITY: low|medium|high"""
        
        prompt = f"""Issues encountered:
{issues_text}

{self._format_context(context) if context else ''}

Diagnose and suggest fixes."""
        
        response = await ollama_client.generate(
            prompt=prompt,
            system=system_prompt,
            temperature=0.3,
            max_tokens=400
        )
        
        # Parse structured response
        diagnosis_dict = self._parse_diagnosis(response)
        return diagnosis_dict
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context dict for AI prompts."""
        lines = []
        if "dataset_rows" in context:
            lines.append(f"Dataset: {context['dataset_rows']} rows")
        if "task_type" in context:
            lines.append(f"Task: {context['task_type']}")
        if "base_model" in context:
            lines.append(f"Model: {context['base_model']}")
        if "gpu_available" in context:
            lines.append(f"GPU: {'Yes' if context['gpu_available'] else 'No'}")
        
        return "\n".join(lines)
    
    def _format_metrics(
        self,
        metrics: Dict[str, Any],
        logs: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Format metrics for AI analysis."""
        text = "Final metrics:\n"
        for key, value in metrics.items():
            if isinstance(value, float):
                text += f"- {key}: {value:.4f}\n"
            else:
                text += f"- {key}: {value}\n"
        
        if logs and len(logs) > 0:
            text += "\nTraining progression:\n"
            # Show first, middle, last logs
            sample_logs = [logs[0]]
            if len(logs) > 2:
                sample_logs.append(logs[len(logs)//2])
            sample_logs.append(logs[-1])
            
            for log in sample_logs:
                if "loss" in log:
                    text += f"  Step {log.get('step', '?')}: loss={log['loss']:.4f}\n"
        
        return text
    
    def _parse_diagnosis(self, response: str) -> Dict[str, Any]:
        """Parse structured diagnosis from AI response."""
        try:
            # Extract sections
            diagnosis = ""
            fixes = []
            severity = "medium"
            
            lines = response.split("\n")
            current_section = None
            
            for line in lines:
                line = line.strip()
                if line.startswith("DIAGNOSIS:"):
                    diagnosis = line.replace("DIAGNOSIS:", "").strip()
                    current_section = "diagnosis"
                elif line.startswith("FIXES:"):
                    current_section = "fixes"
                elif line.startswith("SEVERITY:"):
                    severity_text = line.replace("SEVERITY:", "").strip().lower()
                    severity = severity_text if severity_text in ["low", "medium", "high"] else "medium"
                    current_section = None
                elif current_section == "fixes" and line.startswith("-"):
                    fixes.append(line[1:].strip())
                elif current_section == "diagnosis" and line:
                    diagnosis += " " + line
            
            return {
                "diagnosis": diagnosis.strip() or response[:200],
                "suggested_fixes": fixes or ["Review training logs for details"],
                "severity": severity
            }
        except Exception as e:
            logger.warning("Failed to parse diagnosis", error=str(e))
            return {
                "diagnosis": response[:200],
                "suggested_fixes": ["Review the AI response for details"],
                "severity": "medium"
            }


# Global instance
ai_assistant = AIAssistant()
