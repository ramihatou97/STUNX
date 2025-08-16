/**
 * MediShift Frontend Type Definitions
 * Version: 4.0 - Complete Implementation with Backend Alignment
 */

export * from '../../shared/types';
export * from '../../shared/types/cross-month-post-call';

import { Timestamp } from 'firebase/firestore';

// Additional frontend-specific types

export interface UIState {
  selectedView: 'calendar' | 'list' | 'analytics';
  filters: ScheduleFilters;
  sortBy: string;
  sortOrder: 'asc' | 'desc';
  pageSize: number;
  currentPage: number;
}

export interface ScheduleFilters {
  residents?: string[];
  dateRange?: {
    start: Date;
    end: Date;
  };
  assignmentTypes?: string[];
  teams?: ('Red' | 'Blue')[];
  showConflicts?: boolean;
  showOptimizations?: boolean;
}

export interface NotificationPreferences {
  email: {
    enabled: boolean;
    frequency: 'immediate' | 'daily' | 'weekly';
    types: string[];
  };
  push: {
    enabled: boolean;
    types: string[];
  };
  sms: {
    enabled: boolean;
    types: string[];
  };
}

export interface DragDropContext {
  sourceId: string;
  targetId: string;
  assignmentType: string;
  date: Date;
}

export interface ScheduleValidation {
  isValid: boolean;
  violations: Array<{
    type: string;
    severity: 'warning' | 'error';
    message: string;
    affectedAssignments: string[];
  }>;
  suggestions: string[];
}

export interface ExportOptions {
  format: 'pdf' | 'excel' | 'csv' | 'ical';
  includeDetails: boolean;
  includeStats: boolean;
  dateRange?: {
    start: Date;
    end: Date;
  };
  residents?: string[];
}

export interface ImportResult {
  success: boolean;
  importedCount: number;
  skippedCount: number;
  errors: Array<{
    row: number;
    field: string;
    message: string;
  }>;
  warnings: string[];
}

export interface ScheduleSnapshot {
  id: string;
  createdAt: Timestamp;
  createdBy: string;
  description: string;
  scheduleData: any;
  metadata: {
    type: 'yearly' | 'monthly' | 'weekly';
    period: string;
    residentCount: number;
    assignmentCount: number;
  };
}

export interface BulkOperation {
  id: string;
  type: 'create' | 'update' | 'delete';
  entityType: string;
  entities: any[];
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  errors?: string[];
  startedAt?: Timestamp;
  completedAt?: Timestamp;
}

// Real-time collaboration types
export interface CollaborationSession {
  id: string;
  scheduleId: string;
  participants: Array<{
    userId: string;
    userName: string;
    role: string;
    joinedAt: Timestamp;
    isActive: boolean;
    cursor?: {
      x: number;
      y: number;
    };
  }>;
  changes: Array<{
    userId: string;
    timestamp: Timestamp;
    action: string;
    details: any;
  }>;
}

// Performance monitoring
export interface PerformanceMetrics {
  loadTime: number;
  renderTime: number;
  apiLatency: number;
  cacheHitRate: number;
  errorRate: number;
  userActions: Array<{
    action: string;
    duration: number;
    timestamp: Timestamp;
  }>;
}

// Type guards with complete implementation
export const isValidSchedule = (schedule: any): boolean => {
  return schedule && 
         schedule.id && 
         Array.isArray(schedule.assignments || schedule.blocks || schedule.days);
};

export const hasScheduleConflict = (
  assignment1: any, 
  assignment2: any
): boolean => {
  if (!assignment1.date || !assignment2.date) return false;
  
  const date1 = assignment1.date.toDate ? assignment1.date.toDate() : new Date(assignment1.date);
  const date2 = assignment2.date.toDate ? assignment2.date.toDate() : new Date(assignment2.date);
  
  return date1.getTime() === date2.getTime() && 
         assignment1.residentId === assignment2.residentId;
};

export const isCompleteAssignment = (assignment: any): boolean => {
  return assignment &&
         assignment.residentId &&
         assignment.date &&
         assignment.type &&
         assignment.status !== 'Cancelled';
};

export const calculateFairnessScore = (
  assignments: any[], 
  residents: any[]
): number => {
  if (!assignments.length || !residents.length) return 0;
  
  const counts = new Map<string, number>();
  residents.forEach(r => counts.set(r.id, 0));
  assignments.forEach(a => {
    const current = counts.get(a.residentId) || 0;
    counts.set(a.residentId, current + 1);
  });
  
  const values = Array.from(counts.values());
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
  const stdDev = Math.sqrt(variance);
  
  // Return normalized fairness score (0-100)
  return Math.max(0, Math.min(100, 100 - (stdDev / mean) * 100));
};