"""
Ollama client for TinyLlama integration.
Provides a typed interface to the Ollama API with health checking and retry logic.
"""
import httpx
from typing import Dict, Any, List, Optional, AsyncIterator
from tenacity import retry, stop_after_attempt, wait_exponential
from app.utils.config import settings
from app.utils.logging import get_logger
from app.utils.exceptions import AgentException

logger = get_logger(__name__)


class OllamaClient:
    """
    Client for Ollama API communication.
    
    Ollama runs in a separate Docker container and serves TinyLlama model.
    This client handles all communication with auto-retry and error handling.
    """
    
    def __init__(self, base_url: str = "http://ollama:11434"):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama service URL (default: Docker service name)
        """
        self.base_url = base_url
        self.model_name = "tinyllama"  # Pre-pulled during Docker build
        self.timeout = 60.0  # Generous timeout for LLM inference
        
        logger.info("Ollama client initialized", base_url=self.base_url, model=self.model_name)
    
    async def health_check(self) -> bool:
        """
        Check if Ollama service is healthy and TinyLlama is available.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                
                if response.status_code != 200:
                    logger.warning("Ollama health check failed", status_code=response.status_code)
                    return False
                
                data = response.json()
                models = data.get("models", [])
                model_names = [m["name"] for m in models]
                
                if self.model_name not in model_names:
                    logger.error(
                        "TinyLlama model not found",
                        available_models=model_names,
                        expected=self.model_name
                    )
                    return False
                
                logger.info("Ollama service healthy", models=model_names)
                return True
        
        except Exception as e:
            logger.error("Ollama health check exception", error=str(e))
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False
    ) -> str:
        """
        Generate text completion using TinyLlama.
        
        Args:
            prompt: User prompt
            system: Optional system prompt for context
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response (not implemented)
        
        Returns:
            Generated text
        
        Raises:
            AgentException: If generation fails after retries
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,  # Always non-streaming for now
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            if system:
                payload["system"] = system
            
            logger.debug("Ollama generate request", prompt_length=len(prompt), temperature=temperature)
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                
                if response.status_code != 200:
                    error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                    logger.error("Ollama generate failed", error=error_msg)
                    raise AgentException("AI_GENERATION_ERROR", error_msg)
                
                data = response.json()
                generated_text = data.get("response", "")
                
                logger.info(
                    "Ollama generate success",
                    prompt_tokens=data.get("prompt_eval_count", 0),
                    response_tokens=data.get("eval_count", 0),
                    response_length=len(generated_text)
                )
                
                return generated_text
        
        except httpx.TimeoutException as e:
            error_msg = f"Ollama request timeout after {self.timeout}s"
            logger.error("Ollama timeout", error=error_msg)
            raise AgentException("AI_TIMEOUT_ERROR", error_msg)
        
        except httpx.ConnectError as e:
            error_msg = f"Cannot connect to Ollama at {self.base_url}"
            logger.error("Ollama connection failed", error=error_msg, base_url=self.base_url)
            raise AgentException("AI_CONNECTION_ERROR", error_msg)
        
        except Exception as e:
            error_msg = f"Ollama generation failed: {str(e)}"
            logger.error("Ollama unexpected error", error=str(e))
            raise AgentException("AI_ERROR", error_msg)
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """
        Chat completion using message history.
        
        Args:
            messages: List of {"role": "user|assistant", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
        
        Returns:
            AI response text
        """
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            logger.debug("Ollama chat request", num_messages=len(messages))
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                
                if response.status_code != 200:
                    error_msg = f"Ollama chat error: {response.status_code} - {response.text}"
                    logger.error("Ollama chat failed", error=error_msg)
                    raise AgentException("AI_CHAT_ERROR", error_msg)
                
                data = response.json()
                message = data.get("message", {})
                content = message.get("content", "")
                
                logger.info("Ollama chat success", response_length=len(content))
                
                return content
        
        except Exception as e:
            error_msg = f"Ollama chat failed: {str(e)}"
            logger.error("Ollama chat error", error=str(e))
            raise AgentException("AI_CHAT_ERROR", error_msg)


# Global Ollama client instance
ollama_client = OllamaClient()
