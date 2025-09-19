/**
 * Simplified API Service for Single-User KOO Platform
 * Basic HTTP client with optional API key authentication
 */

interface ApiConfig {
  baseURL: string;
  apiKey?: string;
}

class ApiService {
  private baseURL: string;
  private apiKey?: string;

  constructor(config: ApiConfig) {
    this.baseURL = config.baseURL;
    this.apiKey = config.apiKey;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    // Add API key if available
    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  // Chapter API methods
  async getChapters(params?: {
    skip?: number;
    limit?: number;
    status?: string;
    search?: string;
  }) {
    const queryParams = new URLSearchParams();
    if (params?.skip !== undefined) queryParams.set('skip', params.skip.toString());
    if (params?.limit !== undefined) queryParams.set('limit', params.limit.toString());
    if (params?.status) queryParams.set('status', params.status);
    if (params?.search) queryParams.set('search', params.search);

    const query = queryParams.toString();
    return this.request(`/api/v1/chapters/${query ? `?${query}` : ''}`);
  }

  async getChapter(id: number) {
    return this.request(`/api/v1/chapters/${id}`);
  }

  async createChapter(chapter: {
    title: string;
    content: string;
    summary?: string;
    tags?: string[];
    specialty?: string;
  }) {
    return this.request(`/api/v1/chapters/`, {
      method: 'POST',
      body: JSON.stringify(chapter),
    });
  }

  async updateChapter(id: number, updates: {
    title?: string;
    content?: string;
    summary?: string;
    tags?: string[];
    specialty?: string;
    status?: string;
  }) {
    return this.request(`/api/v1/chapters/${id}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
  }

  async deleteChapter(id: number) {
    return this.request(`/api/v1/chapters/${id}`, {
      method: 'DELETE',
    });
  }

  async getChapterStats(id: number) {
    return this.request(`/api/v1/chapters/${id}/stats`);
  }

  async duplicateChapter(id: number) {
    return this.request(`/api/v1/chapters/${id}/duplicate`, {
      method: 'POST',
    });
  }

  // Research API methods
  async searchResearch(query: {
    query: string;
    max_results?: number;
    date_range?: [number, number];
    journal_filter?: string;
  }) {
    return this.request(`/api/v1/research/search`, {
      method: 'POST',
      body: JSON.stringify(query),
    });
  }

  async getSavedSearches() {
    return this.request(`/api/v1/research/searches`);
  }

  async synthesizeResearch(data: {
    sources: any[];
    topic: string;
    focus_areas?: string[];
  }) {
    return this.request(`/api/v1/research/synthesize`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getResearchTrends(params?: {
    specialty?: string;
    timeframe?: string;
  }) {
    const queryParams = new URLSearchParams();
    if (params?.specialty) queryParams.set('specialty', params.specialty);
    if (params?.timeframe) queryParams.set('timeframe', params.timeframe);

    const query = queryParams.toString();
    return this.request(`/api/v1/research/trends${query ? `?${query}` : ''}`);
  }

  async getResearchRecommendations() {
    return this.request(`/api/v1/research/recommendations`);
  }

  // Admin API methods
  async getAdminInfo() {
    return this.request(`/api/v1/admin/info`);
  }

  async getAPIKeys() {
    return this.request(`/api/v1/admin/api-keys`);
  }

  async addAPIKey(provider: string, key: string, endpointUrl?: string) {
    return this.request(`/api/v1/admin/api-keys`, {
      method: 'POST',
      body: JSON.stringify({
        provider,
        key,
        endpoint_url: endpointUrl
      })
    });
  }

  async removeAPIKey(provider: string) {
    return this.request(`/api/v1/admin/api-keys/${provider}`, {
      method: 'DELETE'
    });
  }

  async validateAPIKey(provider: string) {
    return this.request(`/api/v1/admin/api-keys/${provider}/validate`, {
      method: 'POST'
    });
  }

  async validateAllAPIKeys() {
    return this.request(`/api/v1/admin/api-keys/validate-all`, {
      method: 'POST'
    });
  }

  async getAPIUsageStats() {
    return this.request(`/api/v1/admin/api-keys/usage`);
  }

  async exportAPIConfiguration(includeKeys: boolean = false) {
    return this.request(`/api/v1/admin/api-keys/export?include_keys=${includeKeys}`);
  }

  async getAdminHealthCheck() {
    return this.request(`/api/v1/admin/health`);
  }

  // Health check
  async healthCheck() {
    return this.request(`/health`);
  }
}

// Create API instance
const apiService = new ApiService({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  apiKey: process.env.REACT_APP_API_KEY, // Optional for external access
});

export default apiService;