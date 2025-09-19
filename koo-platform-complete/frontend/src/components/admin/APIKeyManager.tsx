/**
 * API Key Management Component
 * Secure interface for managing AI service API keys
 */

import React, { useState, useEffect } from 'react';
import apiService from '../../services/api';

interface APIKeyConfig {
  provider: string;
  is_configured: boolean;
  is_valid: boolean;
  last_validated?: string;
  usage_count: number;
  rate_limit_remaining?: number;
  rate_limit_reset?: string;
  endpoint_url?: string;
}

interface ValidationResult {
  provider: string;
  is_valid: boolean;
  message: string;
  validated_at: string;
}

interface HealthCheck {
  timestamp: string;
  total_providers: number;
  valid_providers: number;
  invalid_providers: number;
  providers: Record<string, any>;
}

const APIProviders = [
  {
    key: 'gemini',
    name: 'Gemini 2.5 Pro',
    description: 'Google\'s advanced AI for deep research',
    icon: '🧠'
  },
  {
    key: 'claude',
    name: 'Claude Opus 4.1',
    description: 'Anthropic\'s AI for extended thinking',
    icon: '🤖'
  },
  {
    key: 'pubmed',
    name: 'PubMed',
    description: 'Medical literature database',
    icon: '📚'
  },
  {
    key: 'perplexity',
    name: 'Perplexity',
    description: 'AI-enhanced search and research',
    icon: '🔍'
  },
  {
    key: 'semantic_scholar',
    name: 'Semantic Scholar',
    description: 'Academic papers and citations',
    icon: '🎓'
  },
  {
    key: 'elsevier',
    name: 'Elsevier',
    description: 'Medical textbooks and journals',
    icon: '📖'
  },
  {
    key: 'biodigital',
    name: 'BioDigital',
    description: 'Anatomical models and visualizations',
    icon: '🫀'
  }
];

const APIKeyManager: React.FC = () => {
  const [apiKeys, setApiKeys] = useState<APIKeyConfig[]>([]);
  const [healthStatus, setHealthStatus] = useState<HealthCheck | null>(null);
  const [loading, setLoading] = useState(true);
  const [validating, setValidating] = useState<string | null>(null);
  const [showAddForm, setShowAddForm] = useState<string | null>(null);
  const [newKey, setNewKey] = useState('');
  const [endpointUrl, setEndpointUrl] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    loadAPIKeys();
    loadHealthStatus();
  }, []);

  const loadAPIKeys = async () => {
    try {
      const response = await apiService.request('/api/v1/admin/api-keys');
      setApiKeys(response);
    } catch (err: any) {
      setError(`Failed to load API keys: ${err.message}`);
    }
  };

  const loadHealthStatus = async () => {
    try {
      const response = await apiService.request('/api/v1/admin/health');
      setHealthStatus(response);
      setLoading(false);
    } catch (err: any) {
      setError(`Failed to load health status: ${err.message}`);
      setLoading(false);
    }
  };

  const addAPIKey = async (provider: string) => {
    if (!newKey.trim()) {
      setError('API key is required');
      return;
    }

    try {
      const response = await apiService.request('/api/v1/admin/api-keys', {
        method: 'POST',
        body: JSON.stringify({
          provider,
          key: newKey,
          endpoint_url: endpointUrl || undefined
        })
      });

      setSuccess(`API key for ${provider} added successfully`);
      setNewKey('');
      setEndpointUrl('');
      setShowAddForm(null);
      await loadAPIKeys();
      await loadHealthStatus();
    } catch (err: any) {
      setError(`Failed to add API key: ${err.message}`);
    }
  };

  const removeAPIKey = async (provider: string) => {
    if (!confirm(`Are you sure you want to remove the API key for ${provider}?`)) {
      return;
    }

    try {
      await apiService.request(`/api/v1/admin/api-keys/${provider}`, {
        method: 'DELETE'
      });

      setSuccess(`API key for ${provider} removed successfully`);
      await loadAPIKeys();
      await loadHealthStatus();
    } catch (err: any) {
      setError(`Failed to remove API key: ${err.message}`);
    }
  };

  const validateAPIKey = async (provider: string) => {
    setValidating(provider);
    try {
      const response = await apiService.request(`/api/v1/admin/api-keys/${provider}/validate`, {
        method: 'POST'
      });

      if (response.is_valid) {
        setSuccess(`${provider} API key is valid`);
      } else {
        setError(`${provider} API key validation failed: ${response.message}`);
      }

      await loadAPIKeys();
      await loadHealthStatus();
    } catch (err: any) {
      setError(`Failed to validate ${provider}: ${err.message}`);
    } finally {
      setValidating(null);
    }
  };

  const validateAllKeys = async () => {
    setValidating('all');
    try {
      const response = await apiService.request('/api/v1/admin/api-keys/validate-all', {
        method: 'POST'
      });

      setSuccess(`Validated ${response.valid_count}/${response.total_providers} API keys`);
      await loadAPIKeys();
      await loadHealthStatus();
    } catch (err: any) {
      setError(`Failed to validate all keys: ${err.message}`);
    } finally {
      setValidating(null);
    }
  };

  const getStatusColor = (isValid: boolean, isConfigured: boolean) => {
    if (!isConfigured) return 'text-gray-500';
    return isValid ? 'text-green-600' : 'text-red-600';
  };

  const getStatusIcon = (isValid: boolean, isConfigured: boolean) => {
    if (!isConfigured) return '⚪';
    return isValid ? '✅' : '❌';
  };

  const formatLastValidated = (dateString?: string) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleString();
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="bg-white shadow-lg rounded-lg overflow-hidden">
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-4">
          <h2 className="text-2xl font-bold text-white">API Key Management</h2>
          <p className="text-blue-100 mt-1">Manage your AI service integrations</p>
        </div>

        {/* Health Status Dashboard */}
        {healthStatus && (
          <div className="p-6 bg-gray-50 border-b">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
              <div className="bg-white p-4 rounded-lg shadow">
                <div className="text-2xl font-bold text-gray-900">{healthStatus.total_providers}</div>
                <div className="text-sm text-gray-600">Total Providers</div>
              </div>
              <div className="bg-white p-4 rounded-lg shadow">
                <div className="text-2xl font-bold text-green-600">{healthStatus.valid_providers}</div>
                <div className="text-sm text-gray-600">Valid Keys</div>
              </div>
              <div className="bg-white p-4 rounded-lg shadow">
                <div className="text-2xl font-bold text-red-600">{healthStatus.invalid_providers}</div>
                <div className="text-sm text-gray-600">Invalid Keys</div>
              </div>
              <div className="bg-white p-4 rounded-lg shadow">
                <div className="text-2xl font-bold text-blue-600">
                  {((healthStatus.valid_providers / healthStatus.total_providers) * 100).toFixed(0)}%
                </div>
                <div className="text-sm text-gray-600">Health Score</div>
              </div>
            </div>

            <div className="flex gap-2">
              <button
                onClick={validateAllKeys}
                disabled={validating === 'all'}
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg disabled:opacity-50"
              >
                {validating === 'all' ? 'Validating...' : 'Validate All Keys'}
              </button>
              <button
                onClick={() => { loadAPIKeys(); loadHealthStatus(); }}
                className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg"
              >
                Refresh
              </button>
            </div>
          </div>
        )}

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

        {/* API Keys Grid */}
        <div className="p-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {APIProviders.map((provider) => {
              const config = apiKeys.find(k => k.provider === provider.key);
              const isConfigured = config?.is_configured || false;
              const isValid = config?.is_valid || false;

              return (
                <div key={provider.key} className="border rounded-lg p-4 bg-white shadow-sm">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-2xl">{provider.icon}</span>
                      <div>
                        <h3 className="font-semibold text-gray-900">{provider.name}</h3>
                        <p className="text-sm text-gray-600">{provider.description}</p>
                      </div>
                    </div>
                    <span className="text-xl">{getStatusIcon(isValid, isConfigured)}</span>
                  </div>

                  <div className={`text-sm font-medium mb-2 ${getStatusColor(isValid, isConfigured)}`}>
                    {!isConfigured ? 'Not Configured' : isValid ? 'Valid' : 'Invalid'}
                  </div>

                  {config && (
                    <div className="text-xs text-gray-500 space-y-1 mb-3">
                      <div>Usage: {config.usage_count} requests</div>
                      <div>Last validated: {formatLastValidated(config.last_validated)}</div>
                      {config.rate_limit_remaining && (
                        <div>Rate limit: {config.rate_limit_remaining} remaining</div>
                      )}
                    </div>
                  )}

                  <div className="flex gap-2">
                    {!isConfigured ? (
                      <button
                        onClick={() => setShowAddForm(provider.key)}
                        className="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-sm py-2 px-3 rounded"
                      >
                        Add Key
                      </button>
                    ) : (
                      <>
                        <button
                          onClick={() => validateAPIKey(provider.key)}
                          disabled={validating === provider.key}
                          className="flex-1 bg-green-600 hover:bg-green-700 text-white text-sm py-2 px-3 rounded disabled:opacity-50"
                        >
                          {validating === provider.key ? 'Validating...' : 'Validate'}
                        </button>
                        <button
                          onClick={() => removeAPIKey(provider.key)}
                          className="bg-red-600 hover:bg-red-700 text-white text-sm py-2 px-3 rounded"
                        >
                          Remove
                        </button>
                      </>
                    )}
                  </div>

                  {/* Add Key Form */}
                  {showAddForm === provider.key && (
                    <div className="mt-4 p-3 bg-gray-50 rounded">
                      <input
                        type="password"
                        placeholder="Enter API key"
                        value={newKey}
                        onChange={(e) => setNewKey(e.target.value)}
                        className="w-full p-2 border rounded mb-2 text-sm"
                      />
                      <input
                        type="url"
                        placeholder="Custom endpoint URL (optional)"
                        value={endpointUrl}
                        onChange={(e) => setEndpointUrl(e.target.value)}
                        className="w-full p-2 border rounded mb-3 text-sm"
                      />
                      <div className="flex gap-2">
                        <button
                          onClick={() => addAPIKey(provider.key)}
                          className="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-sm py-2 px-3 rounded"
                        >
                          Save
                        </button>
                        <button
                          onClick={() => {
                            setShowAddForm(null);
                            setNewKey('');
                            setEndpointUrl('');
                          }}
                          className="bg-gray-600 hover:bg-gray-700 text-white text-sm py-2 px-3 rounded"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default APIKeyManager;