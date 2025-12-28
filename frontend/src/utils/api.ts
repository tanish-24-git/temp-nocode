import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Projects API
export const projectsAPI = {
  create: (data: { project_name: string; description?: string; tags?: string[] }) =>
    api.post('/api/v1/projects', data),
  
  list: (limit = 50, offset = 0) =>
    api.get(`/api/v1/projects?limit=${limit}&offset=${offset}`),
  
  get: (projectId: string) =>
    api.get(`/api/v1/projects/${projectId}`),
  
  update: (projectId: string, data: Partial<{ project_name: string; description: string; tags: string[] }>) =>
    api.patch(`/api/v1/projects/${projectId}`, data),
  
  delete: (projectId: string) =>
    api.delete(`/api/v1/projects/${projectId}`),
};

// Tasks API
export const tasksAPI = {
  suggest: (data: { samples: any[]; column_names: string[]; domain?: string; language?: string }) =>
    api.post('/api/v1/tasks/suggest', data),
  
  preset: (data: { training_mode: 'fast' | 'balanced' | 'high_quality'; dataset_stats: any; gpu_count: number; task_type?: string }) =>
    api.post('/api/v1/tasks/preset', data),
  
  analyzeCSV: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/api/v1/tasks/analyze-csv', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
};

// AI API
export const aiAPI = {
  health: () => api.get('/api/v1/ai/health'),
  
  detectTask: (data: { samples: any[]; column_names: string[] }) =>
    api.post('/api/v1/ai/detect-task', data),
  
  suggestConfig: (data: { dataset_stats: any; task_type?: string; gpu_count: number }) =>
    api.post('/api/v1/ai/suggest-config', data),
  
  explainMetrics: (data: { metrics: any; training_logs?: any[] }) =>
    api.post('/api/v1/ai/explain-metrics', data),
  
  diagnoseErrors: (data: { errors: string[]; warnings: string[]; context?: any }) =>
    api.post('/api/v1/ai/diagnose-errors', data),
  
  chat: (data: { message: string; context?: any }) =>
    api.post('/api/v1/ai/chat', data),
};

// Jobs API
export const jobsAPI = {
  submit: (pipelineConfig: any) =>
    api.post('/api/v1/jobs', { pipeline_config: pipelineConfig }),
  
  get: (jobId: string) =>
    api.get(`/api/v1/jobs/${jobId}`),
  
  cancel: (jobId: string) =>
    api.delete(`/api/v1/jobs/${jobId}`),
};

// Datasets API
export const datasetsAPI = {
  upload: (file: File, datasetName: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('dataset_name', datasetName);
    return api.post('/api/v1/datasets/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  
  get: (datasetId: string) =>
    api.get(`/api/v1/datasets/${datasetId}`),
  
  preview: (datasetId: string) =>
    api.get(`/api/v1/datasets/${datasetId}/preview`),
};

// Models API
export const modelsAPI = {
  list: () => api.get('/api/v1/models'),
  
  get: (modelId: string) =>
    api.get(`/api/v1/models/${modelId}`),
  
  download: (modelId: string) =>
    api.get(`/api/v1/models/${modelId}/download`, { responseType: 'blob' }),
  
  card: (modelId: string) =>
    api.get(`/api/v1/models/${modelId}/card`),
};

// Health check
export const healthCheck = () => api.get('/health');
