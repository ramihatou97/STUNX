/**
 * Hybrid AI Services Manager
 * Interface for managing API/Web hybrid access to AI services
 */

import React, { useState, useEffect } from 'react';
import apiService from '../../services/api';

interface ServiceStatus {
  service: string;
  api_available: boolean;
  web_available: boolean;
  current_method: string;
  daily_budget: number;
  budget_used: number;
  budget_remaining: number;
  api_calls_today: number;
  web_calls_today: number;
  last_used?: string;
}

interface AIQueryRequest {
  service: string;
  prompt: string;
  max_tokens?: number;
  temperature?: number;
}

interface AIQueryResponse {
  service: string;
  prompt: string;
  response: string;
  method_used: string;
  tokens_used?: number;
  cost?: number;
  timestamp: string;
}

const AIServices = [
  {
    key: 'gemini',
    name: 'Gemini 2.5 Pro',
    description: 'Deep research and analysis',
    icon: '🧠',
    color: 'bg-blue-500'
  },
  {
    key: 'claude',
    name: 'Claude Opus 4.1',
    description: 'Extended thinking and synthesis',
    icon: '🤖',
    color: 'bg-purple-500'
  },
  {
    key: 'perplexity',
    name: 'Perplexity',
    description: 'Real-time AI search',
    icon: '🔍',
    color: 'bg-green-500'
  }
];

const HybridAIManager: React.FC = () => {
  const [services, setServices] = useState<Record<string, ServiceStatus>>({});
  const [loading, setLoading] = useState(true);
  const [testingService, setTestingService] = useState<string | null>(null);
  const [queryService, setQueryService] = useState<string | null>(null);
  const [testPrompt, setTestPrompt] = useState('Explain the basics of neurosurgery in one paragraph.');
  const [queryResult, setQueryResult] = useState<AIQueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [browserInitialized, setBrowserInitialized] = useState(false);

  useEffect(() => {
    loadServicesStatus();
  }, []);

  const loadServicesStatus = async () => {
    try {
      const response = await apiService.request('/api/v1/ai/services/status');
      setServices(response.services);
      setLoading(false);
    } catch (err: any) {
      setError(`Failed to load services status: ${err.message}`);
      setLoading(false);
    }
  };

  const testService = async (service: string) => {
    setTestingService(service);
    setError(null);

    try {
      const response = await apiService.request(`/api/v1/ai/services/${service}/test`, {
        method: 'POST'
      });

      if (response.test_successful) {
        setSuccess(`${service} test successful!`);
      } else {
        setError(`${service} test failed: ${response.error}`);
      }

      await loadServicesStatus();
    } catch (err: any) {
      setError(`Test failed for ${service}: ${err.message}`);
    } finally {
      setTestingService(null);
    }
  };

  const queryAI = async () => {
    if (!queryService || !testPrompt.trim()) {
      setError('Please select a service and enter a prompt');
      return;
    }

    setTestingService(queryService);
    setError(null);
    setQueryResult(null);

    try {
      const response = await apiService.request('/api/v1/ai/query', {
        method: 'POST',
        body: JSON.stringify({
          service: queryService,
          prompt: testPrompt,
          max_tokens: 500
        })
      });

      setQueryResult(response);
      await loadServicesStatus();
    } catch (err: any) {
      setError(`Query failed: ${err.message}`);
    } finally {
      setTestingService(null);
    }
  };

  const initializeBrowser = async () => {
    try {
      await apiService.request('/api/v1/ai/services/initialize', {
        method: 'POST'
      });
      setBrowserInitialized(true);
      setSuccess('Browser session initialized for web automation');
    } catch (err: any) {
      setError(`Failed to initialize browser: ${err.message}`);
    }
  };

  const cleanupBrowser = async () => {
    try {
      await apiService.request('/api/v1/ai/services/cleanup', {
        method: 'POST'
      });
      setBrowserInitialized(false);
      setSuccess('Browser session cleaned up');
    } catch (err: any) {
      setError(`Failed to cleanup browser: ${err.message}`);
    }
  };

  const resetUsageStats = async () => {
    try {
      await apiService.request('/api/v1/ai/services/reset-usage', {
        method: 'POST'
      });
      setSuccess('Usage statistics reset successfully');
      await loadServicesStatus();
    } catch (err: any) {
      setError(`Failed to reset usage stats: ${err.message}`);
    }
  };

  const getStatusColor = (service: ServiceStatus) => {
    const budgetPercent = (service.budget_used / service.daily_budget) * 100;
    if (budgetPercent >= 90) return 'text-red-600';
    if (budgetPercent >= 70) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getAccessMethodBadge = (service: ServiceStatus) => {
    const hasAPI = service.api_available;
    const hasWeb = service.web_available;

    if (hasAPI && hasWeb) {
      return <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded">Hybrid</span>;
    } else if (hasAPI) {
      return <span className="bg-green-100 text-green-800 text-xs px-2 py-1 rounded">API Only</span>;
    } else if (hasWeb) {
      return <span className="bg-orange-100 text-orange-800 text-xs px-2 py-1 rounded">Web Only</span>;
    }
    return <span className="bg-red-100 text-red-800 text-xs px-2 py-1 rounded">Unavailable</span>;
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="bg-white shadow-lg rounded-lg overflow-hidden">
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-4">
          <h2 className="text-2xl font-bold text-white">Hybrid AI Services Manager</h2>
          <p className="text-blue-100 mt-1">Manage API and web interface access to AI services</p>
        </div>

        {/* Global Controls */}
        <div className="p-6 bg-gray-50 border-b">
          <div className="flex flex-wrap gap-4">
            <button
              onClick={initializeBrowser}
              disabled={browserInitialized}
              className={`px-4 py-2 rounded-lg text-white font-medium ${
                browserInitialized
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700'
              }`}
            >
              {browserInitialized ? 'Browser Initialized' : 'Initialize Browser'}
            </button>

            <button
              onClick={cleanupBrowser}
              disabled={!browserInitialized}
              className={`px-4 py-2 rounded-lg text-white font-medium ${
                !browserInitialized
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-orange-600 hover:bg-orange-700'
              }`}
            >
              Cleanup Browser
            </button>

            <button
              onClick={resetUsageStats}
              className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-medium"
            >
              Reset Daily Stats
            </button>

            <button
              onClick={loadServicesStatus}
              className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg font-medium"
            >
              Refresh Status
            </button>
          </div>
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

        {/* Services Grid */}
        <div className="p-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            {AIServices.map((aiService) => {
              const service = services[aiService.key];
              if (!service) return null;

              const budgetPercent = (service.budget_used / service.daily_budget) * 100;

              return (
                <div key={aiService.key} className="border rounded-lg p-6 bg-white shadow-sm">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className={`w-12 h-12 ${aiService.color} rounded-lg flex items-center justify-center`}>
                        <span className="text-2xl">{aiService.icon}</span>
                      </div>
                      <div>
                        <h3 className="font-bold text-gray-900">{aiService.name}</h3>
                        <p className="text-sm text-gray-600">{aiService.description}</p>
                      </div>
                    </div>
                    {getAccessMethodBadge(service)}
                  </div>

                  {/* Budget Usage */}
                  <div className="mb-4">
                    <div className="flex justify-between text-sm mb-1">
                      <span>Daily Budget</span>
                      <span className={getStatusColor(service)}>
                        ${service.budget_used.toFixed(2)} / ${service.daily_budget.toFixed(2)}
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          budgetPercent >= 90 ? 'bg-red-500' :
                          budgetPercent >= 70 ? 'bg-yellow-500' : 'bg-green-500'
                        }`}
                        style={{ width: `${Math.min(budgetPercent, 100)}%` }}
                      ></div>
                    </div>
                  </div>

                  {/* Usage Stats */}
                  <div className="grid grid-cols-2 gap-4 mb-4 text-sm">
                    <div className="text-center p-2 bg-blue-50 rounded">
                      <div className="font-bold text-blue-900">{service.api_calls_today}</div>
                      <div className="text-blue-600">API Calls</div>
                    </div>
                    <div className="text-center p-2 bg-orange-50 rounded">
                      <div className="font-bold text-orange-900">{service.web_calls_today}</div>
                      <div className="text-orange-600">Web Calls</div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2">
                    <button
                      onClick={() => testService(aiService.key)}
                      disabled={testingService === aiService.key}
                      className="flex-1 bg-green-600 hover:bg-green-700 text-white text-sm py-2 px-3 rounded disabled:opacity-50"
                    >
                      {testingService === aiService.key ? 'Testing...' : 'Test'}
                    </button>
                    <button
                      onClick={() => setQueryService(aiService.key)}
                      className="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-sm py-2 px-3 rounded"
                    >
                      Query
                    </button>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Query Interface */}
          <div className="bg-gray-50 rounded-lg p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4">Test AI Query</h3>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Service
                </label>
                <select
                  value={queryService || ''}
                  onChange={(e) => setQueryService(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Choose a service</option>
                  {AIServices.map(service => (
                    <option key={service.key} value={service.key}>
                      {service.name}
                    </option>
                  ))}
                </select>

                <label className="block text-sm font-medium text-gray-700 mb-2 mt-4">
                  Prompt
                </label>
                <textarea
                  value={testPrompt}
                  onChange={(e) => setTestPrompt(e.target.value)}
                  rows={4}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter your prompt here..."
                />

                <button
                  onClick={queryAI}
                  disabled={!queryService || !testPrompt.trim() || testingService !== null}
                  className="w-full mt-4 bg-blue-600 hover:bg-blue-700 text-white py-3 px-4 rounded-lg font-medium disabled:opacity-50"
                >
                  {testingService ? 'Processing...' : 'Send Query'}
                </button>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Response
                </label>
                <div className="h-64 p-3 border border-gray-300 rounded-lg bg-white overflow-y-auto">
                  {queryResult ? (
                    <div>
                      <div className="flex justify-between items-center mb-3 pb-2 border-b">
                        <span className="font-medium text-gray-900">{queryResult.service}</span>
                        <span className="text-xs text-gray-500">
                          {queryResult.method_used} | {queryResult.tokens_used} tokens
                        </span>
                      </div>
                      <div className="text-gray-800 whitespace-pre-wrap">
                        {queryResult.response}
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-400">
                      Response will appear here...
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HybridAIManager;