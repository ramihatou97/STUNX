/**
 * Knowledge Integration Pipeline Interface
 * Complete interface for managing research workflows and knowledge synthesis
 */

import React, { useState, useEffect } from 'react';
import apiService from '../../services/api';

interface ResearchQuery {
  topic: string;
  neurosurgical_focus: string[];
  mesh_terms: string[];
  date_range?: string[];
  max_results: number;
  quality_threshold: number;
}

interface PipelineExecution {
  execution_id: string;
  topic: string;
  stage: string;
  started_at: string;
  completed_at?: string;
  success?: boolean;
  results_count: number;
  has_synthesis?: boolean;
  error?: string;
  progress_percentage?: number;
}

interface KnowledgeSynthesis {
  topic: string;
  executive_summary: string;
  key_findings: string[];
  clinical_implications: string[];
  surgical_techniques: string[];
  evidence_quality: string;
  conflicting_findings: string[];
  research_gaps: string[];
  recommendations: string[];
  sources_used: string[];
  confidence_score: number;
  last_updated: string;
}

interface ResearchTemplate {
  name: string;
  template: ResearchQuery;
}

const NeurosurgicalFocusOptions = [
  'brain tumors',
  'spinal surgery',
  'vascular neurosurgery',
  'epilepsy surgery',
  'functional neurosurgery',
  'pediatric neurosurgery',
  'trauma neurosurgery',
  'minimally invasive surgery',
  'stereotactic surgery',
  'neuromonitoring',
  'brain mapping',
  'cranial surgery',
  'tumor resection',
  'deep brain stimulation'
];

const KnowledgePipeline: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'create' | 'monitor' | 'history' | 'synthesis'>('create');
  const [query, setQuery] = useState<ResearchQuery>({
    topic: '',
    neurosurgical_focus: [],
    mesh_terms: [],
    max_results: 50,
    quality_threshold: 0.7
  });

  const [activeExecutions, setActiveExecutions] = useState<PipelineExecution[]>([]);
  const [executionHistory, setExecutionHistory] = useState<PipelineExecution[]>([]);
  const [selectedExecution, setSelectedExecution] = useState<string | null>(null);
  const [synthesis, setSynthesis] = useState<KnowledgeSynthesis | null>(null);
  const [templates, setTemplates] = useState<ResearchTemplate[]>([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    loadActiveExecutions();
    loadExecutionHistory();
    loadTemplates();

    // Set up polling for active executions
    const interval = setInterval(() => {
      if (activeExecutions.length > 0) {
        loadActiveExecutions();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const loadActiveExecutions = async () => {
    try {
      const response = await apiService.request('/api/v1/knowledge/pipeline/active');
      setActiveExecutions(response.executions);
    } catch (err: any) {
      console.error('Failed to load active executions:', err);
    }
  };

  const loadExecutionHistory = async () => {
    try {
      const response = await apiService.request('/api/v1/knowledge/pipeline/history?limit=10');
      setExecutionHistory(response.history);
    } catch (err: any) {
      console.error('Failed to load execution history:', err);
    }
  };

  const loadTemplates = async () => {
    try {
      const response = await apiService.request('/api/v1/knowledge/pipeline/templates');
      setTemplates(response.templates);
    } catch (err: any) {
      console.error('Failed to load templates:', err);
    }
  };

  const startPipeline = async () => {
    if (!query.topic.trim() || query.neurosurgical_focus.length === 0) {
      setError('Please provide a topic and select at least one neurosurgical focus area');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await apiService.request('/api/v1/knowledge/pipeline/start', {
        method: 'POST',
        body: JSON.stringify(query)
      });

      setSuccess(`Pipeline started successfully! Execution ID: ${response.execution_id}`);
      setActiveTab('monitor');
      await loadActiveExecutions();

      // Reset form
      setQuery({
        topic: '',
        neurosurgical_focus: [],
        mesh_terms: [],
        max_results: 50,
        quality_threshold: 0.7
      });

    } catch (err: any) {
      setError(`Failed to start pipeline: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const loadSynthesis = async (executionId: string) => {
    setLoading(true);
    try {
      const response = await apiService.request(`/api/v1/knowledge/pipeline/${executionId}/synthesis`);
      setSynthesis(response);
      setSelectedExecution(executionId);
      setActiveTab('synthesis');
    } catch (err: any) {
      setError(`Failed to load synthesis: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const cancelExecution = async (executionId: string) => {
    try {
      await apiService.request(`/api/v1/knowledge/pipeline/${executionId}/cancel`, {
        method: 'POST'
      });
      setSuccess('Pipeline execution cancelled');
      await loadActiveExecutions();
      await loadExecutionHistory();
    } catch (err: any) {
      setError(`Failed to cancel execution: ${err.message}`);
    }
  };

  const applyTemplate = (template: ResearchTemplate) => {
    setQuery(template.template);
    setSuccess(`Applied template: ${template.name}`);
  };

  const handleFocusChange = (focus: string, checked: boolean) => {
    if (checked) {
      setQuery(prev => ({
        ...prev,
        neurosurgical_focus: [...prev.neurosurgical_focus, focus]
      }));
    } else {
      setQuery(prev => ({
        ...prev,
        neurosurgical_focus: prev.neurosurgical_focus.filter(f => f !== focus)
      }));
    }
  };

  const getStageColor = (stage: string) => {
    const colors = {
      'initiated': 'text-blue-600',
      'research_gathering': 'text-yellow-600',
      'analysis': 'text-orange-600',
      'synthesis': 'text-purple-600',
      'quality_check': 'text-indigo-600',
      'chapter_update': 'text-green-600',
      'completed': 'text-green-700',
      'failed': 'text-red-600'
    };
    return colors[stage] || 'text-gray-600';
  };

  const formatStage = (stage: string) => {
    return stage.split('_').map(word =>
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="bg-white shadow-lg rounded-lg overflow-hidden">
        <div className="bg-gradient-to-r from-green-600 to-blue-600 px-6 py-4">
          <h2 className="text-2xl font-bold text-white">Knowledge Integration Pipeline</h2>
          <p className="text-green-100 mt-1">Automated research synthesis and knowledge generation</p>
        </div>

        {/* Navigation Tabs */}
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6">
            {[
              { key: 'create', label: 'Create Pipeline', icon: '🚀' },
              { key: 'monitor', label: 'Monitor', icon: '📊' },
              { key: 'history', label: 'History', icon: '📋' },
              { key: 'synthesis', label: 'Synthesis', icon: '🧠' }
            ].map(tab => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key as any)}
                className={`py-4 px-2 border-b-2 font-medium text-sm ${
                  activeTab === tab.key
                    ? 'border-green-500 text-green-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {tab.icon} {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Error/Success Messages */}
        {error && (
          <div className="p-4 bg-red-50 border-l-4 border-red-400">
            <div className="flex">
              <div className="text-red-700">{error}</div>
              <button onClick={() => setError(null)} className="ml-auto text-red-400 hover:text-red-600">×</button>
            </div>
          </div>
        )}

        {success && (
          <div className="p-4 bg-green-50 border-l-4 border-green-400">
            <div className="flex">
              <div className="text-green-700">{success}</div>
              <button onClick={() => setSuccess(null)} className="ml-auto text-green-400 hover:text-green-600">×</button>
            </div>
          </div>
        )}

        {/* Tab Content */}
        <div className="p-6">
          {/* Create Pipeline Tab */}
          {activeTab === 'create' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Research Query Form */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-gray-900">Research Configuration</h3>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Research Topic *
                    </label>
                    <textarea
                      value={query.topic}
                      onChange={(e) => setQuery(prev => ({ ...prev, topic: e.target.value }))}
                      rows={3}
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
                      placeholder="e.g., Latest advances in minimally invasive brain tumor surgery"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Neurosurgical Focus Areas * (Select multiple)
                    </label>
                    <div className="grid grid-cols-2 gap-2 max-h-48 overflow-y-auto border border-gray-200 rounded-lg p-3">
                      {NeurosurgicalFocusOptions.map(focus => (
                        <label key={focus} className="flex items-center space-x-2 text-sm">
                          <input
                            type="checkbox"
                            checked={query.neurosurgical_focus.includes(focus)}
                            onChange={(e) => handleFocusChange(focus, e.target.checked)}
                            className="rounded border-gray-300 text-green-600 focus:ring-green-500"
                          />
                          <span className="capitalize">{focus}</span>
                        </label>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Additional MeSH Terms (Optional)
                    </label>
                    <input
                      type="text"
                      value={query.mesh_terms.join(', ')}
                      onChange={(e) => setQuery(prev => ({
                        ...prev,
                        mesh_terms: e.target.value.split(',').map(t => t.trim()).filter(t => t)
                      }))}
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
                      placeholder="e.g., Intraoperative Monitoring, Brain Mapping"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Max Results
                      </label>
                      <input
                        type="number"
                        value={query.max_results}
                        onChange={(e) => setQuery(prev => ({ ...prev, max_results: parseInt(e.target.value) }))}
                        min="10"
                        max="200"
                        className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Quality Threshold
                      </label>
                      <input
                        type="number"
                        value={query.quality_threshold}
                        onChange={(e) => setQuery(prev => ({ ...prev, quality_threshold: parseFloat(e.target.value) }))}
                        min="0.0"
                        max="1.0"
                        step="0.1"
                        className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
                      />
                    </div>
                  </div>

                  <button
                    onClick={startPipeline}
                    disabled={loading || !query.topic.trim() || query.neurosurgical_focus.length === 0}
                    className="w-full bg-green-600 hover:bg-green-700 text-white py-3 px-4 rounded-lg font-medium disabled:opacity-50"
                  >
                    {loading ? 'Starting Pipeline...' : 'Start Knowledge Pipeline'}
                  </button>
                </div>

                {/* Templates */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-gray-900">Quick Start Templates</h3>
                  <div className="space-y-3">
                    {templates.map((template, index) => (
                      <div key={index} className="border border-gray-200 rounded-lg p-4">
                        <h4 className="font-medium text-gray-900 mb-2">{template.name}</h4>
                        <p className="text-sm text-gray-600 mb-3">{template.template.topic}</p>
                        <div className="flex flex-wrap gap-1 mb-3">
                          {template.template.neurosurgical_focus.map(focus => (
                            <span key={focus} className="inline-block bg-green-100 text-green-800 text-xs px-2 py-1 rounded">
                              {focus}
                            </span>
                          ))}
                        </div>
                        <button
                          onClick={() => applyTemplate(template)}
                          className="bg-blue-600 hover:bg-blue-700 text-white text-sm py-2 px-3 rounded"
                        >
                          Use Template
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Monitor Tab */}
          {activeTab === 'monitor' && (
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold text-gray-900">Active Pipeline Executions</h3>
                <button
                  onClick={loadActiveExecutions}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg"
                >
                  Refresh
                </button>
              </div>

              {activeExecutions.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  No active pipeline executions
                </div>
              ) : (
                <div className="space-y-4">
                  {activeExecutions.map(execution => (
                    <div key={execution.execution_id} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex justify-between items-start mb-3">
                        <div>
                          <h4 className="font-medium text-gray-900">{execution.topic}</h4>
                          <p className="text-sm text-gray-600">ID: {execution.execution_id}</p>
                        </div>
                        <div className="text-right">
                          <span className={`font-medium ${getStageColor(execution.stage)}`}>
                            {formatStage(execution.stage)}
                          </span>
                          <p className="text-sm text-gray-500">
                            {execution.results_count} results
                          </p>
                        </div>
                      </div>

                      {execution.progress_percentage !== undefined && (
                        <div className="mb-3">
                          <div className="flex justify-between text-sm mb-1">
                            <span>Progress</span>
                            <span>{execution.progress_percentage}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-green-500 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${execution.progress_percentage}%` }}
                            ></div>
                          </div>
                        </div>
                      )}

                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-500">
                          Started: {new Date(execution.started_at).toLocaleString()}
                        </span>
                        <button
                          onClick={() => cancelExecution(execution.execution_id)}
                          className="bg-red-600 hover:bg-red-700 text-white text-sm py-1 px-3 rounded"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* History Tab */}
          {activeTab === 'history' && (
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold text-gray-900">Pipeline Execution History</h3>
                <button
                  onClick={loadExecutionHistory}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg"
                >
                  Refresh
                </button>
              </div>

              <div className="space-y-3">
                {executionHistory.map(execution => (
                  <div key={execution.execution_id} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <h4 className="font-medium text-gray-900 mb-1">{execution.topic}</h4>
                        <div className="flex items-center space-x-4 text-sm text-gray-600">
                          <span className={getStageColor(execution.stage)}>
                            {formatStage(execution.stage)}
                          </span>
                          <span>{execution.results_count} results</span>
                          <span>
                            {new Date(execution.started_at).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                      <div className="flex space-x-2">
                        {execution.has_synthesis && execution.success && (
                          <button
                            onClick={() => loadSynthesis(execution.execution_id)}
                            className="bg-green-600 hover:bg-green-700 text-white text-sm py-1 px-3 rounded"
                          >
                            View Synthesis
                          </button>
                        )}
                        {execution.error && (
                          <span className="text-red-600 text-sm">Error: {execution.error}</span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Synthesis Tab */}
          {activeTab === 'synthesis' && (
            <div className="space-y-6">
              {synthesis ? (
                <div>
                  <div className="border-b border-gray-200 pb-4 mb-6">
                    <h3 className="text-xl font-bold text-gray-900">{synthesis.topic}</h3>
                    <div className="flex items-center space-x-4 mt-2 text-sm text-gray-600">
                      <span>Confidence Score: {(synthesis.confidence_score * 100).toFixed(1)}%</span>
                      <span>Last Updated: {new Date(synthesis.last_updated).toLocaleString()}</span>
                      <span>{synthesis.sources_used.length} sources</span>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="space-y-6">
                      {/* Executive Summary */}
                      <div>
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">Executive Summary</h4>
                        <div className="bg-blue-50 p-4 rounded-lg">
                          <p className="text-gray-800">{synthesis.executive_summary}</p>
                        </div>
                      </div>

                      {/* Key Findings */}
                      <div>
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">Key Findings</h4>
                        <ul className="space-y-2">
                          {synthesis.key_findings.map((finding, index) => (
                            <li key={index} className="flex items-start space-x-2">
                              <span className="text-green-600 mt-1">•</span>
                              <span className="text-gray-800">{finding}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Clinical Implications */}
                      <div>
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">Clinical Implications</h4>
                        <ul className="space-y-2">
                          {synthesis.clinical_implications.map((implication, index) => (
                            <li key={index} className="flex items-start space-x-2">
                              <span className="text-blue-600 mt-1">•</span>
                              <span className="text-gray-800">{implication}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    <div className="space-y-6">
                      {/* Surgical Techniques */}
                      <div>
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">Surgical Techniques</h4>
                        <ul className="space-y-2">
                          {synthesis.surgical_techniques.map((technique, index) => (
                            <li key={index} className="flex items-start space-x-2">
                              <span className="text-purple-600 mt-1">•</span>
                              <span className="text-gray-800">{technique}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Recommendations */}
                      <div>
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">Recommendations</h4>
                        <ul className="space-y-2">
                          {synthesis.recommendations.map((recommendation, index) => (
                            <li key={index} className="flex items-start space-x-2">
                              <span className="text-orange-600 mt-1">•</span>
                              <span className="text-gray-800">{recommendation}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Research Gaps */}
                      {synthesis.research_gaps.length > 0 && (
                        <div>
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">Research Gaps</h4>
                          <ul className="space-y-2">
                            {synthesis.research_gaps.map((gap, index) => (
                              <li key={index} className="flex items-start space-x-2">
                                <span className="text-red-600 mt-1">•</span>
                                <span className="text-gray-800">{gap}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Evidence Quality */}
                  <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                    <h4 className="text-lg font-semibold text-gray-900 mb-2">Evidence Quality Assessment</h4>
                    <p className="text-gray-800">{synthesis.evidence_quality}</p>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  Select a completed pipeline execution to view its synthesis
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default KnowledgePipeline;