import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Send, Loader2, Bot, User } from 'lucide-react';
import { aiAPI } from '../utils/api';
import toast from 'react-hot-toast';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export default function AIChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await aiAPI.chat({ message: input });
      
      const aiMessage: Message = {
        role: 'assistant',
        content: response.data.response,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'AI chat failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/10 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 text-center"
        >
          <h1 className="text-4xl font-bold gradient-text flex items-center justify-center gap-3">
            <Sparkles className="w-10 h-10" />
            AI Assistant
          </h1>
          <p className="text-muted-foreground mt-2">
            Ask anything about fine-tuning, hyperparameters, or best practices
          </p>
        </motion.div>

        {/* Chat Container */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass rounded-2xl p-6 mb-6 h-[600px] flex flex-col"
        >
          {/* Messages */}
          <div className="flex-1 overflow-y-auto space-y-4 mb-4">
            {messages.length === 0 ? (
              <div className="h-full flex items-center justify-center text-muted-foreground">
                <div className="text-center">
                  <Bot className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p>Start a conversation with the AI assistant</p>
                  <p className="text-sm mt-2">Try: "What batch size for 10k samples?"</p>
                </div>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: msg.role === 'user' ? 20 : -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
                >
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                    msg.role === 'user' 
                      ? 'bg-primary text-primary-foreground' 
                      : 'bg-accent text-accent-foreground'
                  }`}>
                    {msg.role === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
                  </div>
                  
                  <div className={`flex-1 max-w-[80%] ${msg.role === 'user' ? 'text-right' : ''}`}>
                    <div className={`inline-block px-4 py-3 rounded-2xl ${
                      msg.role === 'user'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-secondary text-secondary-foreground'
                    }`}>
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1 px-2">
                      {msg.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </motion.div>
              ))
            )}
            
            {loading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex gap-3"
              >
                <div className="w-10 h-10 rounded-full bg-accent text-accent-foreground flex items-center justify-center">
                  <Bot className="w-5 h-5" />
                </div>
                <div className="flex-1">
                  <div className="inline-block px-4 py-3 rounded-2xl bg-secondary">
                    <Loader2 className="w-5 h-5 animate-spin" />
                  </div>
                </div>
              </motion.div>
            )}
          </div>

          {/* Input */}
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Ask the AI assistant..."
              className="flex-1 bg-secondary border border-border rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary"
              disabled={loading}
            />
            <button
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              className="bg-primary text-primary-foreground px-6 py-3 rounded-xl hover:opacity-90 disabled:opacity-50 transition-all glow-primary"
            >
              {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
            </button>
          </div>
        </motion.div>

        {/* Quick Actions */}
        <div className="grid grid-cols-2 gap-4">
          {[
            'Explain LoRA vs QLoRA',
            'Best learning rate for chat models',
            'When to use 4-bit quantization',
            'How to prevent overfitting'
          ].map((prompt, idx) => (
            <motion.button
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              onClick={() => setInput(prompt)}
              className="glass p-4 rounded-xl hover:bg-secondary transition-all text-left text-sm"
            >
              {prompt}
            </motion.button>
          ))}
        </div>
      </div>
    </div>
  );
}
