import { Link } from 'react-router-dom'
import { ArrowRight, Sparkles, Zap, Shield, Database, Upload, Settings, Download } from 'lucide-react'
import { motion } from 'framer-motion'

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      {/* Hero Section */}
      <section className="pt-20 pb-32 px-4">
        <div className="max-w-6xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 mb-6">
              <Sparkles className="w-4 h-4" />
              <span className="text-sm font-medium">No-Code LLM Fine-Tuning</span>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-gray-900 via-primary-800 to-gray-900 dark:from-white dark:via-primary-400 dark:to-white bg-clip-text text-transparent">
              Fine-Tune LLMs
              <br />
              Without Writing Code
            </h1>
            
            <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-3xl mx-auto">
              Drag-and-drop pipeline builder for training, evaluating, and deploying custom language models.
              Supports LoRA, QLoRA, and full fine-tuning with real-time monitoring.
            </p>
            
            <div className="flex gap-4 justify-center">
              <Link
                to="/playground"
                className="px-8 py-4 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium flex items-center gap-2 transition-all transform hover:scale-105 shadow-lg hover:shadow-xl"
              >
                Start Building
                <ArrowRight className="w-5 h-5" />
              </Link>
              <a
                href="#workflow"
                className="px-8 py-4 bg-white dark:bg-gray-800 border-2 border-gray-300 dark:border-gray-600 hover:border-primary-500 dark:hover:border-primary-500 text-gray-700 dark:text-gray-200 rounded-lg font-medium transition-all"
              >
                See How It Works
              </a>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4 bg-white dark:bg-gray-800/50">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-12 text-gray-900 dark:text-white">
            Powerful Features
          </h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            <FeatureCard
              icon={<Zap className="w-8 h-8" />}
              title="Fast Training"
              description="LoRA and QLoRA for parameter-efficient fine-tuning. Train on CPU or GPU with automatic optimization."
            />
            <FeatureCard
              icon={<Shield className="w-8 h-8" />}
              title="Production Ready"
              description="Docker-based deployment, error handling, and structured logging. Built for scale."
            />
            <FeatureCard
              icon={<Database className="w-8 h-8" />}
              title="Redis-Backed"
              description="Fast metadata storage with Redis. MinIO for model artifacts and checkpoints."
            />
          </div>
        </div>
      </section>

      {/* Workflow Section */}
      <section id="workflow" className="py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-4 text-gray-900 dark:text-white">
            How It Works
          </h2>
          <p className="text-center text-gray-600 dark:text-gray-300 mb-16 max-w-2xl mx-auto">
            Build your fine-tuning pipeline in 4 simple steps
          </p>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            <WorkflowStep
              number="1"
              icon={<Upload className="w-6 h-6" />}
              title="Upload Dataset"
              description="Upload your training data in CSV, JSON, or JSONL format"
            />
            <WorkflowStep
              number="2"
              icon={<Settings className="w-6 h-6" />}
              title="Configure Pipeline"
              description="Drag agents onto canvas: validation → preprocessing → training"
            />
            <WorkflowStep
              number="3"
              icon={<Zap className="w-6 h-6" />}
              title="Train Model"
              description="Choose LoRA, QLoRA, or full fine-tuning. Monitor in real-time"
            />
            <WorkflowStep
              number="4"
              icon={<Download className="w-6 h-6" />}
              title="Export & Deploy"
              description="Download adapter weights, merged model, or GGUF format"
            />
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 bg-gradient-to-r from-primary-600 to-primary-800">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6 text-white">
            Ready to Fine-Tune Your First Model?
          </h2>
          <p className="text-xl text-primary-100 mb-8">
            Start building your pipeline in the playground. No credit card required.
          </p>
          <Link
            to="/playground"
            className="inline-flex items-center gap-2 px-8 py-4 bg-white text-primary-700 rounded-lg font-medium hover:bg-gray-100 transition-all transform hover:scale-105 shadow-lg"
          >
            Open Playground
            <ArrowRight className="w-5 h-5" />
          </Link>
        </div>
      </section>
    </div>
  )
}

function FeatureCard({ icon, title, description }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="p-6 rounded-xl bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:border-primary-500 dark:hover:border-primary-500 transition-all"
    >
      <div className="w-12 h-12 rounded-lg bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 flex items-center justify-center mb-4">
        {icon}
      </div>
      <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">{title}</h3>
      <p className="text-gray-600 dark:text-gray-300">{description}</p>
    </motion.div>
  )
}

function WorkflowStep({ number, icon, title, description }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay: parseInt(number) * 0.1 }}
      className="relative"
    >
      <div className="flex flex-col items-center text-center">
        <div className="w-16 h-16 rounded-full bg-primary-600 text-white flex items-center justify-center text-2xl font-bold mb-4 shadow-lg">
          {number}
        </div>
        <div className="w-12 h-12 rounded-lg bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 flex items-center justify-center mb-4">
          {icon}
        </div>
        <h3 className="text-lg font-semibold mb-2 text-gray-900 dark:text-white">{title}</h3>
        <p className="text-sm text-gray-600 dark:text-gray-300">{description}</p>
      </div>
    </motion.div>
  )
}
