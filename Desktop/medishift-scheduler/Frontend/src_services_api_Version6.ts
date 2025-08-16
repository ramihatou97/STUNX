import { httpsCallable } from 'firebase/functions';
import { functions } from '../firebase-config';

// Typed function calls
export const api = {
  // Schedule generation
  generateYearlySchedule: httpsCallable<
    { academicYearId: string; residents: any[]; config: any },
    { success: boolean; scheduleId: string }
  >(functions, 'generateYearlySchedule'),
  
  generateMonthlySchedule: httpsCallable<
    { month: number; year: number; staffingLevel: 'Normal' | 'Shortage' },
    { success: boolean; scheduleId: string }
  >(functions, 'generateMonthlySchedule'),
  
  generateWeeklySchedule: httpsCallable<
    { weekStartDate: string; residents: any[] },
    { success: boolean; scheduleId: string }
  >(functions, 'generateWeeklySchedule'),
  
  // Analytics
  generateAnalyticsReport: httpsCallable<
    { period: 'weekly' | 'monthly' | 'quarterly' },
    { success: boolean; reportId: string; reportData: any }
  >(functions, 'generateAnalyticsReport'),
  
  // Admin functions
  setAdminClaim: httpsCallable<
    { uid: string; isAdmin: boolean },
    { success: boolean; message: string }
  >(functions, 'setAdminClaim'),
};

// Error handling wrapper
export async function callFunction<T, R>(
  fn: (data: T) => Promise<{ data: R }>,
  data: T
): Promise<R> {
  try {
    const result = await fn(data);
    return result.data;
  } catch (error: any) {
    console.error('Function call failed:', error);
    throw new Error(error.message || 'Function call failed');
  }
}