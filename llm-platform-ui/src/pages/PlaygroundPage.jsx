import { useState } from 'react'
import { Play, Upload, Database, CheckCircle, Settings, Zap, Download, Plus } from 'lucide-react'
import toast, { Toaster } from 'react-hot-toast'

const AGENT_TYPES = [
  { id: 'dataset', name: 'Dataset', icon: Database, color: 'bg-blue-500', description: 'Load training data' },
  { id: 'validation', name: 'Validation', icon: CheckCircle, color: 'bg-green-500', description: 'Validate dataset' },
  { id: 'preprocessing', name: 'Preprocessing', icon: Settings, color: 'bg-purple-500', description: 'Clean & tokenize' },
  { id: 'training', name: 'Training', icon: Zap, color: 'bg-orange-500', description: 'Fine-tune model' },
  { id: 'evaluation', name: 'Evaluation', icon: CheckCircle, color: 'bg-teal-500', description: 'Compute metrics' },
  { id: 'export', name: 'Export', icon: Download, color: 'bg-pink-500', description: 'Export model' },
]

export default function PlaygroundPage() {
  const [pipeline, setPipeline] = useState([])
  const [selectedAgent, setSelectedAgent] = useState(null)
  const [config, setConfig] = useState({})

  const addAgent = (agentType) => {
    const newAgent = {
      id: `${agentType.id}_${Date.now()}`,
      type: agentType.id,
      name: agentType.name,
      icon: agentType.icon,
      color: agentType.color,
      config: {}
    }
    setPipeline([...pipeline, newAgent])
    toast.success(`Added ${agentType.name} agent`)
  }

  const removeAgent = (agentId) => {
    setPipeline(pipeline.filter(a => a.id !== agentId))
    if (selectedAgent?.id === agentId) {
      setSelectedAgent(null)
    }
    toast.success('Agent removed')
  }

  const runPipeline = async () => {
    if (pipeline.length === 0) {
      toast.error('Add at least one agent to the pipeline')
      return
    }

    toast.loading('Submitting pipeline job...')
    
    // TODO: Call API to submit job
    setTimeout(() => {
      toast.dismiss()
      toast.success('Pipeline job submitted!')
    }, 1000)
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-6">
      <Toaster position="top-right" />
      
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Pipeline Playground
          </h1>
          <p className="text-gray-600 dark:text-gray-300">
            Drag and drop agents to build your fine-tuning pipeline
          </p>
        </div>

        <div className="grid lg:grid-cols-4 gap-6">
          {/* Agent Palette */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 sticky top-24">
              <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
                Available Agents
              </h2>
              <div className="space-y-3">
                {AGENT_TYPES.map((agent) => (
                  <button
                    key={agent.id}
                    onClick={() => addAgent(agent)}
                    className="w-full p-3 rounded-lg border-2 border-gray-200 dark:border-gray-700 hover:border-primary-500 dark:hover:border-primary-500 transition-all text-left group"
                  >
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-lg ${agent.color} flex items-center justify-center`}>
                        <agent.icon className="w-5 h-5 text-white" />
                      </div>
                      <div className="flex-1">
                        <div className="font-medium text-gray-900 dark:text-white group-hover:text-primary-600 dark:group-hover:text-primary-400">
                          {agent.name}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {agent.description}
                        </div>
                      </div>
                      <Plus className="w-4 h-4 text-gray-400 group-hover:text-primary-600 dark:group-hover:text-primary-400" />
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Pipeline Canvas */}
          <div className="lg:col-span-2">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 min-h-[600px]">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Pipeline ({pipeline.length} agents)
                </h2>
                <button
                  onClick={runPipeline}
                  disabled={pipeline.length === 0}
                  className="px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white rounded-lg font-medium flex items-center gap-2 transition-all"
                >
                  <Play className="w-4 h-4" />
                  Run Pipeline
                </button>
              </div>

              {pipeline.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-96 text-center">
                  <div className="w-20 h-20 rounded-full bg-gray-100 dark:bg-gray-700 flex items-center justify-center mb-4">
                    <Upload className="w-10 h-10 text-gray-400" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                    No agents yet
                  </h3>
                  <p className="text-gray-500 dark:text-gray-400 max-w-sm">
                    Add agents from the palette on the left to start building your pipeline
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {pipeline.map((agent, index) => (
                    <div key={agent.id}>
                      <AgentCard
                        agent={agent}
                        index={index}
                        onRemove={() => removeAgent(agent.id)}
                        onClick={() => setSelectedAgent(agent)}
                        isSelected={selectedAgent?.id === agent.id}
                      />
                      {index < pipeline.length - 1 && (
                        <div className="flex justify-center py-2">
                          <div className="w-0.5 h-8 bg-gray-300 dark:bg-gray-600"></div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Configuration Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 sticky top-24">
              <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
                Configuration
              </h2>
              {selectedAgent ? (
                <div>
                  <div className="flex items-center gap-3 mb-4 p-3 rounded-lg bg-gray-50 dark:bg-gray-700">
                    <div className={`w-10 h-10 rounded-lg ${selectedAgent.color} flex items-center justify-center`}>
                      <selectedAgent.icon className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <div className="font-medium text-gray-900 dark:text-white">
                        {selectedAgent.name}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        Agent Configuration
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    {selectedAgent.type === 'training' && (
                      <>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            Training Method
                          </label>
                          <select className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white">
                            <option>LoRA</option>
                            <option>QLoRA</option>
                            <option>Full Fine-tune</option>
                          </select>
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            Epochs
                          </label>
                          <input
                            type="number"
                            defaultValue={3}
                            className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            Batch Size
                          </label>
                          <input
                            type="number"
                            defaultValue={4}
                            className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                          />
                        </div>
                      </>
                    )}
                    {selectedAgent.type === 'dataset' && (
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Dataset File
                        </label>
                        <input
                          type="file"
                          className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        />
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="text-center py-12">
                  <Settings className="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-3" />
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Select an agent to configure
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function AgentCard({ agent, index, onRemove, onClick, isSelected }) {
  return (
    <div
      onClick={onClick}
      className={`p-4 rounded-lg border-2 transition-all cursor-pointer ${
        isSelected
          ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
          : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="text-sm font-medium text-gray-500 dark:text-gray-400">
            #{index + 1}
          </div>
          <div className={`w-10 h-10 rounded-lg ${agent.color} flex items-center justify-center`}>
            <agent.icon className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="font-medium text-gray-900 dark:text-white">
              {agent.name}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              {agent.type}
            </div>
          </div>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation()
            onRemove()
          }}
          className="text-gray-400 hover:text-red-500 transition-colors"
        >
          ×
        </button>
      </div>
    </div>
  )
}
