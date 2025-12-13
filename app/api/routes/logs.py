"""
Logs API routes for real-time log streaming.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from app.infra.logging_stream import LogStream
from app.infra.redis import redis_client
from app.utils.logging import get_logger
import json

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/logs", tags=["logs"])


async def log_event_generator(run_id: str) -> AsyncGenerator[str, None]:
    """Generate Server-Sent Events for log streaming."""
    log_stream = LogStream(redis_client.client)
    
    try:
        async for log_event in log_stream.consume_logs(run_id, start_id="0"):
            if log_event.get("type") == "keepalive":
                yield f": keepalive\n\n"
            else:
                yield f"data: {json.dumps(log_event)}\n\n"
    except Exception as e:
        logger.error("Log streaming error", run_id=run_id, error=str(e))
        yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"


@router.get("/stream/{run_id}")
async def stream_logs(run_id: str):
    """
    Stream real-time logs via Server-Sent Events (SSE).
    
    Client usage:
    ```javascript
    const eventSource = new EventSource('/api/v1/logs/stream/abc123');
    eventSource.onmessage = (event) => {
        const log = JSON.parse(event.data);
        console.log(log);
    };
    ```
    """
    return StreamingResponse(
        log_event_generator(run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/history/{run_id}")
async def get_log_history(run_id: str, limit: int = 100):
    """Get historical logs for a run."""
    try:
        log_stream = LogStream(redis_client.client)
        logs = await log_stream.get_log_history(run_id, count=min(limit, 1000))
        
        return {
            "run_id": run_id,
            "logs": logs,
            "total": len(logs)
        }
    
    except Exception as e:
        logger.error("Failed to get log history", run_id=run_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
