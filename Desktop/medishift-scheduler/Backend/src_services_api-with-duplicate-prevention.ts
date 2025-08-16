import { httpsCallable } from 'firebase/functions';
import { functions } from '../firebase-config';
import { requestManager } from './request-manager';
import axios from 'axios';
import toast from 'react-hot-toast';

class APIService {
  private baseURL: string;
  
  constructor() {
    this.baseURL = process.env.REACT_APP_API_BASE_URL || '';
  }
  
  /**
   * Generate Yearly Schedule with duplicate prevention
   */
  async generateYearlySchedule(params: {
    academicYearId: string;
    residents: any[];
    config: any;
  }) {
    return requestManager.executeRequest(
      '/schedule/yearly/generate',
      'POST',
      async () => {
        const generateFn = httpsCallable(functions, 'generateYearlySchedule');
        const result = await generateFn(params);
        return result.data;
      },
      params,
      {
        allowDuplicate: false,
        minInterval: 5000 // 5 seconds minimum between yearly generation
      }
    );
  }
  
  /**
   * Generate Monthly Schedule with duplicate prevention
   */
  async generateMonthlySchedule(params: {
    month: number;
    year: number;
    staffingLevel: 'Normal' | 'Shortage';
    crossMonthPostCalls?: any[];
    useMLOptimization?: boolean;
  }) {
    return requestManager.executeRequest(
      '/schedule/monthly/generate',
      'POST',
      async () => {
        const generateFn = httpsCallable(functions, 'generateMonthlySchedule');
        const result = await generateFn(params);
        return result.data;
      },
      params,
      {
        allowDuplicate: false,
        minInterval: 3000 // 3 seconds minimum between monthly generation
      }
    );
  }
  
  /**
   * Generate Weekly Schedule with duplicate prevention
   */
  async generateWeeklySchedule(params: {
    weekStartDate: string;
    residents: string[];
    useMLOptimization?: boolean;
  }) {
    return requestManager.executeRequest(
      '/schedule/weekly/generate',
      'POST',
      async () => {
        const generateFn = httpsCallable(functions, 'generateWeeklySchedule');
        const result = await generateFn(params);
        return result.data;
      },
      params,
      {
        allowDuplicate: false,
        minInterval: 2000 // 2 seconds minimum between weekly generation
      }
    );
  }
  
  /**
   * Approve Leave Request with duplicate prevention
   */
  async approveLeaveRequest(requestId: string) {
    return requestManager.executeRequest(
      `/leave/${requestId}/approve`,
      'POST',
      async () => {
        const response = await axios.post(`${this.baseURL}/leave/${requestId}/approve`);
        return response.data;
      },
      { requestId },
      {
        allowDuplicate: false,
        customKey: `approve-leave-${requestId}`,
        minInterval: 5000 // Prevent double approval
      }
    );
  }
  
  /**
   * Submit EPA Assessment with duplicate prevention
   */
  async submitEPAAssessment(epaId: string, assessment: any) {
    return requestManager.executeRequest(
      `/epa/${epaId}/submit`,
      'POST',
      async () => {
        const response = await axios.post(`${this.baseURL}/epa/${epaId}/submit`, assessment);
        return response.data;
      },
      { epaId, assessment },
      {
        allowDuplicate: false,
        customKey: `submit-epa-${epaId}`,
        minInterval: 10000 // 10 seconds to prevent duplicate submissions
      }
    );
  }
  
  /**
   * Get schedule (can be called multiple times)
   */
  async getSchedule(scheduleId: string) {
    // Read operations don't need duplicate prevention
    const response = await axios.get(`${this.baseURL}/schedule/${scheduleId}`);
    return response.data;
  }
}

export const protectedApi = new APIService();