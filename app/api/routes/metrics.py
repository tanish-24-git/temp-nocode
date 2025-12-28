"""
Prometheus metrics endpoint for monitoring.
Exposes custom ML metrics and system metrics for Grafana dashboards.
"""
from fastapi import APIRouter, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])


# Prometheus Metrics

# API Metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'status']
)

# Training Metrics
training_jobs_total = Counter(
    'training_jobs_total',
    'Total training jobs submitted',
    ['status']  # pending, running, completed, failed
)

training_throughput = Gauge(
    'training_throughput_samples_per_sec',
    'Training throughput in samples/sec',
    ['job_id']
)

training_loss = Gauge(
    'training_loss',
    'Current training loss',
    ['job_id', 'epoch']
)

# GPU Metrics (custom, requires nvidia-smi)
gpu_utilization = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

gpu_memory_used = Gauge(
    'gpu_memory_used_bytes',
    'GPU memory used in bytes',
    ['gpu_id']
)

# Queue Metrics
queue_depth = Gauge(
    'queue_depth',
    'Number of pending jobs in queue',
    ['queue']  # training, evaluation, orchestration, default
)

# SSE/Logs Metrics
sse_log_latency = Histogram(
    'sse_log_latency_ms',
    'SSE log streaming latency in milliseconds',
    buckets=[10, 50, 100, 200, 500, 1000]
)

# OOM Kills
oom_kills_total = Counter(
    'oom_kills_total',
    'Total OOM kills (out of memory)',
    ['container']
)


@router.get("")
async def get_metrics():
    """
    Prometheus metrics endpoint.
    
    Exposes all custom metrics for scraping by Prometheus.
    Configure Prometheus to scrape this endpoint:
    
    ```yaml
    scrape_configs:
      - job_name: 'llm-platform-api'
        static_configs:
          - targets: ['api:8000']
        metrics_path: '/api/v1/metrics'
    ```
    """
    try:
        # Generate latest metrics
        metrics_output = generate_latest()
        
        return Response(
            content=metrics_output,
            media_type=CONTENT_TYPE_LATEST
        )
    
    except Exception as e:
        logger
.error("Metrics generation failed", error=str(e))
        return Response(
            content=f"# Error generating metrics: {str(e)}\n",
            media_type="text/plain",
            status_code=500
        )


@router.get("/health")
async def metrics_health():
    """Health check for metrics endpoint."""
    return {
        "status": "healthy",
        "metrics_enabled": True,
        "endpoint": "/api/v1/metrics"
    }


# Helper functions for updating metrics (called by other modules)

def increment_api_request(endpoint: str, status: int):
    """Increment API request counter."""
    api_requests_total.labels(endpoint=endpoint, status=str(status)).inc()


def increment_training_job(status: str):
    """Increment training job counter."""
    training_jobs_total.labels(status=status).inc()


def update_training_metrics(job_id: str, epoch: int, loss: float, throughput: float):
    """Update training metrics."""
    training_loss.labels(job_id=job_id, epoch=str(epoch)).set(loss)
    training_throughput.labels(job_id=job_id).set(throughput)


def update_gpu_metrics(gpu_id: int, utilization: float, memory_used: int):
    """Update GPU metrics."""
    gpu_utilization.labels(gpu_id=str(gpu_id)).set(utilization)
    gpu_memory_used.labels(gpu_id=str(gpu_id)).set(memory_used)


def update_queue_depth(queue_name: str, depth: int):
    """Update queue depth gauge."""
    queue_depth.labels(queue=queue_name).set(depth)


def record_oom_kill(container: str):
    """Record an OOM kill event."""
    oom_kills_total.labels(container=container).inc()


def record_sse_latency(latency_ms: float):
    """Record SSE latency."""
    sse_log_latency.observe(latency_ms)
