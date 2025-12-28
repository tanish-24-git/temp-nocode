---
description: Build and deploy the complete LLM platform
---

# Build and Deploy Workflow

## Prerequisites Check

1. Verify Docker and Docker Compose are installed
   ```bash
   docker --version
   docker-compose --version
   ```

2. Check for GPU (optional - system works on CPU too)
   ```bash
   nvidia-smi
   ```

## Phase 1: Environment Setup

// turbo
3. Copy environment template
   ```bash
   cp .env.example .env
   ```

4. Review and update `.env` file with your settings (especially GPU settings)
   - Set `GPU_ENABLED=true` if you have GPU, `false` for CPU only
   - Set `CUDA_VISIBLE_DEVICES=0` to use first GPU

## Phase 2: Build Docker Images

5. Build all Docker images (this may take 10-15 minutes)
   ```bash
   docker-compose build
   ```

   **Note**: The Ollama image will pre-pull TinyLlama during build (~5-10 min)

## Phase 3: Start Services

// turbo
6. Start all services in background
   ```bash
   docker-compose up -d
   ```

7. Wait for services to be healthy (30-60 seconds)
   ```bash
   sleep 60
   ```

## Phase 4: Verification

// turbo
8. Check service health
   ```bash
   docker-compose ps
   ```

   All services should show "Up (healthy)"

// turbo
9. Test API health endpoint
   ```bash
   curl http://localhost:8000/health
   ```

   Expected: `{"status":"healthy",...}`

// turbo
10. Verify Ollama has TinyLlama pre-pulled
    ```bash
    docker exec llm-platform-ollama ollama list
    ```

    Expected: Should show `tinyllama:latest`

// turbo
11. Check Prometheus metrics endpoint
    ```bash
    curl http://localhost:8000/metrics
    ```

## Phase 5: Access UI

12. Open browser to API docs
    - Swagger UI: http://localhost:8000/docs
    - Frontend: http://localhost:3000
    - MinIO Console: http://localhost:9001 (admin/admin)
    - Grafana: http://localhost:3000 (admin/admin) if monitoring enabled

## Troubleshooting

If any service fails:

13. Check logs for failed service
    ```bash
    docker-compose logs [service_name]
    ```

14. Common issues:
    - **Port conflicts**: Stop services using ports 8000, 3000, 6379, 9000
    - **Ollama not starting**: Check GPU drivers if GPU_ENABLED=true
    - **Worker fails**: Verify GPU settings match your hardware

15. Restart specific service
    ```bash
    docker-compose restart [service_name]
    ```

## Clean Restart

16. If you need to start fresh
    ```bash
    docker-compose down -v
    docker-compose up -d
    ```

## Success Criteria

✅ All 7+ containers running and healthy  
✅ API returns 200 on `/health`  
✅ Ollama has TinyLlama model  
✅ Frontend loads successfully  
✅ Can access Swagger docs
