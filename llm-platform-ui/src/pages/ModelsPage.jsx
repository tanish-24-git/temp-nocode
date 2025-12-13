import { Download, Calendar, Cpu } from 'lucide-react'

export default function ModelsPage() {
  const models = [
    {
      id: 'model_001',
      name: 'llama2-finetuned',
      base_model: 'meta-llama/Llama-2-7b-hf',
      created_at: '2025-12-13T07:30:00Z',
      metrics: { accuracy: 0.92, loss: 0.42 },
      exports: ['adapter', 'merged']
    },
    {
      id: 'model_002',
      name: 'gpt2-custom',
      base_model: 'gpt2',
      created_at: '2025-12-12T15:20:00Z',
      metrics: { accuracy: 0.88, loss: 0.55 },
      exports: ['adapter']
    },
  ]

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Trained Models
          </h1>
          <p className="text-gray-600 dark:text-gray-300">
            Browse and download your fine-tuned models
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {models.map((model) => (
            <div
              key={model.id}
              className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow"
            >
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-1">
                    {model.name}
                  </h3>
                  <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300">
                    <Cpu className="w-4 h-4" />
                    {model.base_model}
                  </div>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
                  <Calendar className="w-4 h-4" />
                  {new Date(model.created_at).toLocaleDateString()}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-700">
                  <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                    Accuracy
                  </div>
                  <div className="text-lg font-semibold text-gray-900 dark:text-white">
                    {(model.metrics.accuracy * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-700">
                  <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                    Final Loss
                  </div>
                  <div className="text-lg font-semibold text-gray-900 dark:text-white">
                    {model.metrics.loss.toFixed(2)}
                  </div>
                </div>
              </div>

              <div className="flex flex-wrap gap-2 mb-4">
                {model.exports.map((format) => (
                  <span
                    key={format}
                    className="px-2 py-1 text-xs font-medium rounded-full bg-primary-100 text-primary-700 dark:bg-primary-900/30 dark:text-primary-400"
                  >
                    {format}
                  </span>
                ))}
              </div>

              <button className="w-full px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium flex items-center justify-center gap-2 transition-colors">
                <Download className="w-4 h-4" />
                Download Model
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
