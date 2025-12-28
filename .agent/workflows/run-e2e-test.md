---
description: Run end-to-end test with AI assistant
---

# End-to-End Test with AI Assistant

This workflow tests the complete pipeline from dataset upload to model export with AI assistance.

## Prerequisites

1. Ensure all Docker services are running
   ```bash
   docker-compose ps
   ```

## Phase 1: Prepare Test Data

// turbo
2. Create a sample CSV dataset
   ```bash
   echo "question,answer
What is AI?,Artificial Intelligence is...
What is ML?,Machine Learning is...
What is DL?,Deep Learning is..." > test_dataset.csv
   ```

## Phase 2: Upload Dataset

3. Upload dataset via API
   ```bash
   curl -X POST http://localhost:8000/api/v1/datasets/upload \
     -F "file=@test_dataset.csv" \
     -F "dataset_name=test_qa"
   ```

   Save the `dataset_id` from response

## Phase 3: AI Task Detection

4. Detect task type using AI
   ```bash
   curl -X POST http://localhost:8000/api/v1/ai/detect-task \
     -H "Content-Type: application/json" \
     -d '{
       "dataset_id": "DATASET_ID_FROM_STEP_3",
       "sample_size": 10
     }'
   ```

   Expected: `{"task_type": "qa", "confidence": 0.9}`

## Phase 4: AI Config Suggestion

5. Get AI-suggested training configuration
   ```bash
   curl -X POST http://localhost:8000/api/v1/ai/suggest-config \
     -H "Content-Type: application/json" \
     -d '{
       "dataset_id": "DATASET_ID_FROM_STEP_3",
       "task_type": "qa",
       "gpu_count": 1
     }'
   ```

   Save the suggested `batch_size`, `epochs`, `lora_rank` from response

## Phase 5: Submit Training Job

6. Submit training job with AI-suggested params
   ```bash
   curl -X POST http://localhost:8000/api/v1/jobs \
     -H "Content-Type: application/json" \
     -d '{
       "pipeline_config": {
         "run_id": "test_run_001",
         "dataset_id": "DATASET_ID_FROM_STEP_3",
         "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
         "task_type": "qa",
         "training": {
           "method": "lora",
           "epochs": 1,
           "batch_size": 4,
           "lora_rank": 16
         }
       }
     }'
   ```

   Save the `job_id` from response

## Phase 6: Monitor Training

7. Stream real-time logs (in browser or curl)
   - Browser: Open http://localhost:8000/api/v1/logs/stream/test_run_001
   - Or use curl:
   ```bash
   curl -N http://localhost:8000/api/v1/logs/stream/test_run_001
   ```

   You should see live training metrics

// turbo
8. Check job status
   ```bash
   curl http://localhost:8000/api/v1/jobs/JOB_ID_FROM_STEP_6
   ```

## Phase 7: Wait for Completion

9. Poll job status until complete (or wait ~5-10 minutes for small dataset)
   ```bash
   while true; do
     STATUS=$(curl -s http://localhost:8000/api/v1/jobs/JOB_ID | jq -r '.status')
     echo "Status: $STATUS"
     [[ "$STATUS" == "completed" ]] && break
     sleep 30
   done
   ```

## Phase 8: Verify Artifacts

// turbo
10. List trained models
    ```bash
    curl http://localhost:8000/api/v1/models
    ```

    Should show new model from this job

11. Download model card
    ```bash
    curl http://localhost:8000/api/v1/models/MODEL_ID/card -o model_card.json
    ```

## Phase 9: AI Metrics Explanation

12. Ask AI to explain training metrics
    ```bash
    curl -X POST http://localhost:8000/api/v1/ai/explain-metrics \
      -H "Content-Type: application/json" \
      -d '{
        "job_id": "JOB_ID_FROM_STEP_6"
      }'
    ```

    Expected: AI explanation of loss curve, convergence, etc.

## Success Criteria

✅ Dataset uploaded successfully  
✅ AI detected correct task type  
✅ AI suggested reasonable hyperparameters  
✅ Training job started  
✅ Real-time logs streamed  
✅ Training completed without errors  
✅ Model artifacts saved to MinIO  
✅ AI explanation generated

## Cleanup

// turbo
13. Remove test dataset file
    ```bash
    rm test_dataset.csv
    ```
