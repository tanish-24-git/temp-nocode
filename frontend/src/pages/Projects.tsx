import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FolderPlus, Folder, Tag, Trash2, Edit, Plus } from 'lucide-react';
import { projectsAPI } from '../utils/api';
import toast from 'react-hot-toast';

interface Project {
  project_id: string;
  project_name: string;
  description: string | null;
  tags: string[];
  created_at: string;
  job_count: number;
  dataset_count: number;
}

export default function Projects() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [newProject, setNewProject] = useState({ project_name: '', description: '', tags: '' });

  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    try {
      const response = await projectsAPI.list();
      setProjects(response.data);
    } catch (error: any) {
      toast.error('Failed to load projects');
    } finally {
      setLoading(false);
    }
  };

  const createProject = async () => {
    try {
      await projectsAPI.create({
        project_name: newProject.project_name,
        description: newProject.description || undefined,
        tags: newProject.tags.split(',').map(t => t.trim()).filter(Boolean),
      });
      
      toast.success('Project created!');
      setShowCreateDialog(false);
      setNewProject({ project_name: '', description: '', tags: '' });
      loadProjects();
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to create project');
    }
  };

  const deleteProject = async (projectId: string) => {
    if (!confirm('Delete this project?')) return;
    
    try {
      await projectsAPI.delete(projectId);
      toast.success('Project deleted');
      loadProjects();
    } catch (error: any) {
      toast.error('Failed to delete project');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/10 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <h1 className="text-4xl font-bold gradient-text flex items-center gap-3">
              <Folder className="w-10 h-10" />
              Projects
            </h1>
            <p className="text-muted-foreground mt-2">
              Organize your fine-tuning workflows
            </p>
          </motion.div>

          <motion.button
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            onClick={() => setShowCreateDialog(true)}
            className="bg-primary text-primary-foreground px-6 py-3 rounded-xl hover:opacity-90 transition-all glow-primary flex items-center gap-2"
          >
            <Plus className="w-5 h-5" />
            New Project
          </motion.button>
        </div>

        {/* Projects Grid */}
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="glass rounded-2xl p-6 h-48 skeleton" />
            ))}
          </div>
        ) : projects.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="glass rounded-2xl p-12 text-center"
          >
            <FolderPlus className="w-24 h-24 mx-auto mb-6 text-muted-foreground opacity-50" />
            <h3 className="text-2xl font-semibold mb-2">No projects yet</h3>
            <p className="text-muted-foreground mb-6">Create your first project to get started</p>
            <button
              onClick={() => setShowCreateDialog(true)}
              className="bg-primary text-primary-foreground px-8 py-3 rounded-xl hover:opacity-90 transition-all"
            >
              Create Project
            </button>
          </motion.div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {projects.map((project, idx) => (
              <motion.div
                key={project.project_id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="glass rounded-2xl p-6 hover:bg-secondary/50 transition-all cursor-pointer group"
              >
                <div className="flex items-start justify-between mb-4">
                  <Folder className="w-8 h-8 text-primary" />
                  <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button className="p-2 hover:bg-secondary rounded-lg">
                      <Edit className="w-4 h-4" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteProject(project.project_id);
                      }}
                      className="p-2 hover:bg-destructive/20 text-destructive rounded-lg"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                <h3 className="text-xl font-semibold mb-2">{project.project_name}</h3>
                
                {project.description && (
                  <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
                    {project.description}
                  </p>
                )}

                <div className="flex flex-wrap gap-2 mb-4">
                  {project.tags.map((tag, i) => (
                    <span
                      key={i}
                      className="inline-flex items-center gap-1 px-2 py-1 bg-accent/20 text-accent text-xs rounded-md"
                    >
                      <Tag className="w-3 h-3" />
                      {tag}
                    </span>
                  ))}
                </div>

                <div className="flex gap-6 text-sm text-muted-foreground">
                  <div>
                    <span className="font-semibold text-foreground">{project.job_count}</span> jobs
                  </div>
                  <div>
                    <span className="font-semibold text-foreground">{project.dataset_count}</span> datasets
                  </div>
                </div>

                <div className="mt-4 pt-4 border-t border-border text-xs text-muted-foreground">
                  Created {new Date(project.created_at).toLocaleDateString()}
                </div>
              </motion.div>
            ))}
          </div>
        )}

        {/* Create Dialog */}
        {showCreateDialog && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-6"
            onClick={() => setShowCreateDialog(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              onClick={(e) => e.stopPropagation()}
              className="glass rounded-2xl p-8 max-w-md w-full"
            >
              <h2 className="text-2xl font-bold mb-6">Create New Project</h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Project Name *</label>
                  <input
                    type="text"
                    value={newProject.project_name}
                    onChange={(e) => setNewProject({ ...newProject, project_name: e.target.value })}
                    placeholder="CustomerSupportBot"
                    className="w-full bg-secondary border border-border rounded-xl px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Description</label>
                  <textarea
                    value={newProject.description}
                    onChange={(e) => setNewProject({ ...newProject, description: e.target.value })}
                    placeholder="24/7 support agent..."
                    rows={3}
                    className="w-full bg-secondary border border-border rounded-xl px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Tags (comma-separated)</label>
                  <input
                    type="text"
                    value={newProject.tags}
                    onChange={(e) => setNewProject({ ...newProject, tags: e.target.value })}
                    placeholder="finance, chat"
                    className="w-full bg-secondary border border-border rounded-xl px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                </div>
              </div>

              <div className="flex gap-3 mt-6">
                <button
                  onClick={() => setShowCreateDialog(false)}
                  className="flex-1 bg-secondary text-foreground px-4 py-3 rounded-xl hover:bg-secondary/80 transition-all"
                >
                  Cancel
                </button>
                <button
                  onClick={createProject}
                  disabled={!newProject.project_name}
                  className="flex-1 bg-primary text-primary-foreground px-4 py-3 rounded-xl hover:opacity-90 disabled:opacity-50 transition-all glow-primary"
                >
                  Create
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
