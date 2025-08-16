import axios, { AxiosInstance } from 'axios';
import { toast } from 'react-hot-toast';

const ML_SERVICE_URL = process.env.REACT_APP_ML_SERVICE_URL || 'https://ml-service.medishift.app';

class MLService {
  private client: AxiosInstance;
  private cache: Map<string, { data: any; timestamp: number }>;
  private cacheTimeout = 5 * 60 * 1000; // 5 minutes

  constructor() {
    this.client = axios.create({
      baseURL: ML_SERVICE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    });

    this.cache = new Map();

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('authToken');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 503) {
          toast.error('ML service is temporarily unavailable');
        }
        return Promise.reject(error);
      }
    );
  }

  private getCacheKey(endpoint: string, params: any): string {
    return `${endpoint}_${JSON.stringify(params)}`;
  }

  private getFromCache(key: string): any | null {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data;
    }
    this.cache.delete(key);
    return null;
  }

  private setCache(key: string, data: any): void {
    this.cache.set(key, { data, timestamp: Date.now() });
  }

  /**
   * Extract schedule from document using OCR/NLP
   */
  async extractSchedule(file: File): Promise<{
    success: boolean;
    data: any;
    confidence: number;
    warnings?: string[];
  }> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('extractionMode', 'comprehensive');
    
    const response = await this.client.post('/extract', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  }
  
  /**
   * Get schedule predictions with enhanced analytics
   */
  async getPredictions(
    startDate: string, 
    periodDays: number = 30
  ): Promise<{
    predictions: any[];
    highRiskDates: string[];
    recommendations: string[];
    confidence: number;
  }> {
    const cacheKey = this.getCacheKey('predict', { startDate, periodDays });
    const cached = this.getFromCache(cacheKey);
    if (cached) return cached;

    const response = await this.client.post('/predict', {
      start_date: startDate,
      period_days: periodDays,
      include_recommendations: true,
      confidence_threshold: 0.8
    });
    
    this.setCache(cacheKey, response.data);
    return response.data;
  }
  
  /**
   * Detect anomalies in schedule with detailed analysis
   */
  async detectAnomalies(scheduleData: any): Promise<{
    hasAnomalies: boolean;
    count: number;
    anomalies: Array<{
      type: string;
      severity: 'low' | 'medium' | 'high' | 'critical';
      description: string;
      affectedDates: string[];
      suggestedFix?: string;
    }>;
    conflicts?: any[];
  }> {
    const response = await this.client.post('/anomalies', {
      schedule: scheduleData,
      checkTypes: ['coverage', 'fairness', 'compliance', 'patterns'],
      includeConflicts: true
    });
    
    return response.data;
  }
  
  /**
   * Get intelligent recommendations with priority scoring
   */
  async getRecommendations(
    schedule: any, 
    constraints: any
  ): Promise<{
    suggestions: Array<{
      priority: number;
      type: string;
      description: string;
      impact: string;
      implementation?: any;
    }>;
    optimizationScore: number;
    potentialImprovement: number;
  }> {
    const response = await this.client.post('/recommendations', {
      schedule,
      constraints,
      optimization_goals: ['fairness', 'coverage', 'compliance'],
      return_implementation: true
    });
    
    return response.data;
  }

  /**
   * Optimize schedule using AI
   */
  async optimizeSchedule(
    schedule: any,
    optimizationParams: {
      targetFairness?: number;
      minCoverage?: number;
      maxConsecutiveCalls?: number;
      preferenceWeight?: number;
    }
  ): Promise<{
    optimizedSchedule: any;
    improvements: any;
    metrics: any;
  }> {
    const response = await this.client.post('/optimize', {
      schedule,
      params: optimizationParams,
      algorithm: 'genetic_algorithm',
      iterations: 1000
    });
    
    return response.data;
  }

  /**
   * Predict leave request impact
   */
  async predictLeaveImpact(
    leaveRequest: any,
    currentSchedule: any
  ): Promise<{
    coverageImpact: number;
    riskLevel: 'low' | 'medium' | 'high';
    affectedShifts: any[];
    alternativeDates?: string[];
  }> {
    const response = await this.client.post('/predict-leave-impact', {
      leave_request: leaveRequest,
      current_schedule: currentSchedule
    });
    
    return response.data;
  }

  /**
   * Get historical analytics
   */
  async getHistoricalAnalytics(
    startDate: string,
    endDate: string,
    metrics: string[]
  ): Promise<any> {
    const response = await this.client.post('/historical-analytics', {
      start_date: startDate,
      end_date: endDate,
      metrics: metrics || ['calls', 'coverage', 'fairness', 'violations']
    });
    
    return response.data;
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear();
  }
}

export const mlService = new MLService();