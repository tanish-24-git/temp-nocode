"""
Training callbacks for real-time metric streaming.
Publishes training metrics to Redis Stream for live monitoring.
"""
from typing import Dict, Any
from transformers import TrainerCallback
from app.infra.logging_stream import LogStream
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LogStreamCallback(TrainerCallback):
    """
    Callback to stream training metrics to Redis Stream in real-time.
    Publishes metrics every N steps for live monitoring via SSE.
    """
    
    def __init__(self, log_stream: LogStream, run_id: str, agent_name: str = "TrainingAgent"):
        self.log_stream = log_stream
        self.run_id = run_id
        self.agent_name = agent_name
    
    async def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when training logs are generated."""
        if logs:
            await self._publish_metrics(logs, state)
    
    async def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        metrics = {
            "step": state.global_step,
            "epoch": state.epoch if state.epoch is not None else 0,
        }
        
        # Add loss if available
        if hasattr(state, 'log_history') and state.log_history:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log:
                metrics['loss'] = latest_log['loss']
            if 'learning_rate' in latest_log:
                metrics['learning_rate'] = latest_log['learning_rate']
        
        await self.log_stream.publish_log(
            run_id=self.run_id,
            agent=self.agent_name,
            level="METRIC",
            message=f"Training step {state.global_step} completed",
            metadata=metrics
        )
    
    async def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        await self.log_stream.publish_log(
            run_id=self.run_id,
            agent=self.agent_name,
            level="INFO",
            message=f"Epoch {int(state.epoch)} completed",
            metadata={"epoch": int(state.epoch), "step": state.global_step}
        )
    
    async def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics:
            await self.log_stream.publish_log(
                run_id=self.run_id,
                agent=self.agent_name,
                level="METRIC",
                message="Evaluation completed",
                metadata=metrics
            )
    
    async def _publish_metrics(self, logs: Dict[str, Any], state):
        """Publish training metrics to log stream."""
        metrics = {
            "step": state.global_step,
            "epoch": state.epoch if state.epoch is not None else 0,
        }
        
        # Extract relevant metrics
        for key in ['loss', 'learning_rate', 'grad_norm', 'eval_loss', 'eval_accuracy']:
            if key in logs:
                metrics[key] = logs[key]
        
        await self.log_stream.publish_log(
            run_id=self.run_id,
            agent=self.agent_name,
            level="METRIC",
            message="Training metrics",
            metadata=metrics
        )
