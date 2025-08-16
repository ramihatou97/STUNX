import axios from 'axios';

const ML_SERVICE_URL = process.env.REACT_APP_ML_SERVICE_URL || 'https://ml-service-xxxxx.run.app';

export class MLService {
  /**
   * Extract schedule from document
   */
  async extractSchedule(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await axios.post(`${ML_SERVICE_URL}/extract`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  }
  
  /**
   * Get schedule predictions
   */
  async getPredictions(date: string, period: number = 30): Promise<any> {
    const response = await axios.post(`${ML_SERVICE_URL}/predict`, {
      start_date: date,
      period_days: period,
    });
    
    return response.data;
  }
  
  /**
   * Detect anomalies in schedule
   */
  async detectAnomalies(scheduleData: any): Promise<any> {
    const response = await axios.post(`${ML_SERVICE_URL}/anomalies`, {
      schedule: scheduleData,
    });
    
    return response.data;
  }
  
  /**
   * Get intelligent recommendations
   */
  async getRecommendations(schedule: any, constraints: any): Promise<any> {
    const response = await axios.post(`${ML_SERVICE_URL}/recommendations`, {
      schedule,
      constraints,
    });
    
    return response.data;
  }
}