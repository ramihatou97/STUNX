import { 
  Resident, 
  CallAssignment, 
  AppConfiguration, 
  AcademicYear,
  LeaveRequest,
  RotationBlock,
  CallType,
  CrossMonthPostCall,
  StaffingLevel,
  CallStrategy,
  ScheduleAnalytics,
  OptimizationResult
} from '../../../shared/types';
import * as admin from 'firebase-admin';
import { v4 as uuidv4 } from 'uuid';

/**
 * Complete Monthly Call Scheduler with All Features
 * 
 * FEATURES IMPLEMENTED:
 * 
 * HARD CONSTRAINTS (Unbreakable Rules):
 * ✅ Vacation & Leave Protection - Complete exclusion from eligible pool
 * ✅ Post-Call Protection - Ineligible day after 24h/weekend/night (NOT after day calls)
 * ✅ PARO Hard Cap - Absolute max calls based on working days (e.g., 27-29 days = 7 calls)
 * ✅ Maximum 2 Weekends/Rotation - Tracks unique weekend blocks (Fri-Sun) per 28-day block
 * ✅ Mandatory PGY-1 Supervision - Always assigns backup (PGY-2+ for day, PGY-3+ for night/weekend)
 * 
 * INTELLIGENT ASSIGNMENT LOGIC:
 * ✅ Two-Tiered Max Call Target - PGY-based targets with staff shortage mode override
 * ✅ Call Equalization & Seniority - Heavy weighting for fairness, lower priority for seniors
 * ✅ Chief Resident Exemption - Extreme penalty score (-1000) for chiefs
 * 
 * SMART CALL TYPES & STRATEGY:
 * ✅ Intelligent Strategy Determination - Analyzes roster for split call viability
 * ✅ Multiple Call Types - 24h, Day, Night, Weekend, Holiday, Backup, PostCall
 * 
 * ADVANCED FEATURES:
 * ✅ AI Optimization Engine - Simulated annealing with constraint validation
 * ✅ Comprehensive Analytics - Gini coefficient, violation tracking, distribution charts
 * 
 * Version: 2.0.0 - Production Release
 * Last Updated: 2025-01-16
 */

/**
 * Call statistics for tracking resident assignments
 */
interface CallStats {
  totalCalls: number;
  weekendCalls: number;
  holidayCalls: number;
  nightCalls: number;
  dayCalls: number;
  twentyFourHourCalls: number;
  backupCalls: number;
  lastCallDate: Date | null;
  consecutiveDays: number;
  callDates: Date[];
  points: number;
  weekendBlocks: Set<string>;
  callTypeHistory: CallType[];
  workingDaysInRotation: number;
  giniContribution: number;
  currentRotationBlock: string;
}

/**
 * Performance metrics for monitoring
 */
interface PerformanceMetrics {
  startTime: number;
  dayProcessingTimes: number[];
  residentFilterTime: number;
  residentSortTime: number;
  assignmentCreationTime: number;
  totalTime?: number;
  peakMemoryUsage?: number;
  userIdentifier?: string;
  generationTimestamp?: string;
  fairnessScore?: number;
  violationCount?: number;
  optimizationImprovement?: number;
}

/**
 * Scheduling violation tracking
 */
interface ScheduleViolation {
  type: 'hard' | 'soft';
  rule: string;
  residentId: string;
  date: Date;
  description: string;
}

/**
 * Rotation block information (28-day blocks)
 */
interface RotationBlockInfo {
  blockId: string;
  startDate: Date;
  endDate: Date;
  blockNumber: number;
}

/**
 * Configuration for the scheduler
 */
interface SchedulerConfig {
  // PARO Hard Caps - Based on working days
  paroHardCaps: Array<{
    minDays: number;
    maxDays: number;
    maxCalls: number;
  }>;
  
  // PGY-Based Call Ratios (1-in-X)
  pgyCallRatios: Record<number, number>;
  
  // Minimum backup seniority requirements
  minimumBackupSeniority: {
    day: number;
    night: number;
    weekend: number;
  };
  
  // Optimization settings
  enableAIOptimization: boolean;
  optimizationIterations: number;
  targetGiniCoefficient: number;
  
  // Strategy preferences
  preferSplitCallStrategy: boolean;
  
  // Constraint weights for scoring
  weights: {
    callEqualization: number;
    seniority: number;
    chiefExemption: number;
    weekendDistribution: number;
    spacing: number;
    variety: number;
    offServiceNight: number;
  };
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: SchedulerConfig = {
  paroHardCaps: [
    { minDays: 27, maxDays: 29, maxCalls: 7 },
    { minDays: 24, maxDays: 26, maxCalls: 6 },
    { minDays: 20, maxDays: 23, maxCalls: 5 },
    { minDays: 16, maxDays: 19, maxCalls: 4 },
    { minDays: 12, maxDays: 15, maxCalls: 3 },
    { minDays: 8, maxDays: 11, maxCalls: 2 },
    { minDays: 1, maxDays: 7, maxCalls: 1 }
  ],
  pgyCallRatios: {
    1: 4, // PGY-1: 1-in-4
    2: 5, // PGY-2: 1-in-5
    3: 6, // PGY-3: 1-in-6
    4: 6, // PGY-4: 1-in-6
    5: 7, // PGY-5: 1-in-7
    6: 7, // PGY-6: 1-in-7
    7: 8  // Fellows: 1-in-8
  },
  minimumBackupSeniority: {
    day: 2,
    night: 3,
    weekend: 3
  },
  enableAIOptimization: true,
  optimizationIterations: 10000,
  targetGiniCoefficient: 0.15,
  preferSplitCallStrategy: true,
  weights: {
    callEqualization: 100,
    seniority: -10,
    chiefExemption: -1000,
    weekendDistribution: 20,
    spacing: 3,
    variety: 10,
    offServiceNight: 30
  }
};

export class MonthlyCallScheduler {
  private residents: Resident[];
  private config: SchedulerConfig;
  private academicYear: AcademicYear;
  private approvedLeave: LeaveRequest[];
  private month: number;
  private year: number;
  private existingAssignments: CallAssignment[];
  private callStats: Map<string, CallStats>;
  private statsCache: Map<string, CallStats>;
  private debugMode: boolean;
  private generatedAssignments: CallAssignment[] = [];
  private crossMonthPostCalls: CrossMonthPostCall[] = [];
  private metrics: PerformanceMetrics;
  private holidays: Date[] = [];
  private weekends: Date[] = [];
  private failedDays: Date[] = [];
  private residentWorkingDays: Map<string, number> = new Map();
  private violations: ScheduleViolation[] = [];
  private staffingLevel: StaffingLevel = 'Normal';
  private callStrategy: CallStrategy = 'Standard';
  private rotationBlocks: Map<string, RotationBlockInfo> = new Map();
  private currentRotationBlock: RotationBlockInfo | null = null;
  private weekendBlockCache: Map<string, string> = new Map();
  
  constructor(
    residents: Resident[],
    config: AppConfiguration,
    academicYear: AcademicYear,
    approvedLeave: LeaveRequest[],
    month: number,
    year: number,
    existingAssignments: CallAssignment[] = [],
    previousMonthPostCalls: CrossMonthPostCall[] = [],
    debugMode: boolean = false,
    customConfig?: Partial<SchedulerConfig>
  ) {
    this.validateInputs(residents, month, year);
    
    this.residents = [...residents];
    this.config = { ...DEFAULT_CONFIG, ...customConfig };
    this.academicYear = academicYear;
    this.approvedLeave = approvedLeave;
    this.month = month;
    this.year = year;
    this.existingAssignments = existingAssignments;
    this.debugMode = debugMode;
    
    console.log(`🚀 Initializing Monthly Call Scheduler v2.0.0`);
    console.log(`📅 Month: ${this.month + 1}/${this.year}`);
    console.log(`👥 Residents: ${this.residents.length}`);
    console.log(`⚙️ Strategy: ${this.config.preferSplitCallStrategy ? 'Split' : 'Standard'}`);
    
    this.metrics = {
      startTime: Date.now(),
      dayProcessingTimes: [],
      residentFilterTime: 0,
      residentSortTime: 0,
      assignmentCreationTime: 0,
      generationTimestamp: new Date().toISOString()
    };
    
    // Initialize all tracking systems
    this.initializeRotationBlocks();
    this.callStats = new Map();
    this.statsCache = new Map();
    this.initializeCallStats();
    this.precalculateCalendarInfo();
    this.calculateWorkingDays();
    this.identifyWeekendBlocks();
    
    if (previousMonthPostCalls?.length > 0) {
      this.processPreviousMonthPostCalls(previousMonthPostCalls);
    }
  }
  
  /**
   * Initialize rotation blocks (28-day blocks)
   */
  private initializeRotationBlocks(): void {
    const academicYearStart = this.academicYear.startDate.toDate();
    const academicYearEnd = this.academicYear.endDate.toDate();
    
    let blockNumber = 0;
    let currentBlockStart = new Date(academicYearStart);
    
    while (currentBlockStart < academicYearEnd) {
      const blockEnd = new Date(currentBlockStart);
      blockEnd.setDate(currentBlockStart.getDate() + 27); // 28-day block
      
      const blockInfo: RotationBlockInfo = {
        blockId: `block-${this.academicYear.id}-${blockNumber}`,
        startDate: new Date(currentBlockStart),
        endDate: blockEnd,
        blockNumber
      };
      
      // Check if this block overlaps with our month
      const monthStart = new Date(this.year, this.month, 1);
      const monthEnd = new Date(this.year, this.month + 1, 0);
      
      if (blockEnd >= monthStart && currentBlockStart <= monthEnd) {
        this.rotationBlocks.set(blockInfo.blockId, blockInfo);
        
        // Set current block if it contains the first of the month
        if (currentBlockStart <= monthStart && blockEnd >= monthStart) {
          this.currentRotationBlock = blockInfo;
        }
      }
      
      currentBlockStart.setDate(currentBlockStart.getDate() + 28);
      blockNumber++;
    }
    
    console.log(`📋 Identified ${this.rotationBlocks.size} rotation blocks for this month`);
  }
  
  /**
   * Main schedule generation method
   */
  public async generateSchedule(
    staffingLevel: StaffingLevel = 'Normal',
    retryCount: number = 0
  ): Promise<CallAssignment[]> {
    console.log(`\n🔄 Starting schedule generation...`);
    console.log(`📊 Staffing level: ${staffingLevel}`);
    
    const startTime = Date.now();
    this.staffingLevel = staffingLevel;
    
    try {
      const daysInMonth = new Date(this.year, this.month + 1, 0).getDate();
      
      // Determine call strategy based on roster
      this.callStrategy = this.determineCallStrategy();
      console.log(`📋 Using ${this.callStrategy} call strategy`);
      
      // Process each day in the month
      for (let day = 1; day <= daysInMonth; day++) {
        const date = new Date(this.year, this.month, day);
        const dayStartTime = Date.now();
        
        try {
          const dayAssignments = await this.assignCallsForDay(date);
          this.generatedAssignments.push(...dayAssignments);
          this.updateCallStatsForDay(dayAssignments);
          
          this.metrics.dayProcessingTimes.push(Date.now() - dayStartTime);
          
          // Periodic cache sync for performance
          if (day % 5 === 0) {
            this.syncStatsCache();
          }
        } catch (error) {
          console.error(`❌ Failed to assign calls for ${date.toDateString()}:`, error);
          this.failedDays.push(date);
          
          // Record violation
          this.violations.push({
            type: 'hard',
            rule: 'Assignment Failure',
            residentId: '',
            date,
            description: `Failed to assign calls: ${error.message}`
          });
        }
      }
      
      // Generate post-call assignments
      const postCallAssignments = this.generatePostCallAssignments();
      this.generatedAssignments.push(...postCallAssignments);
      
      // Run AI optimization if enabled
      if (this.config.enableAIOptimization) {
        console.log(`\n🤖 Running AI optimization...`);
        const optimized = await this.optimizeSchedule();
        this.generatedAssignments = optimized.schedule;
        this.metrics.optimizationImprovement = optimized.improvements.scoreImprovement;
      }
      
      // Calculate final metrics
      this.calculateFairnessMetrics();
      
      this.metrics.totalTime = Date.now() - startTime;
      
      console.log(`\n✅ Schedule generation complete!`);
      console.log(`📊 Total assignments: ${this.generatedAssignments.length}`);
      console.log(`⏱️ Total time: ${this.metrics.totalTime}ms`);
      console.log(`📈 Fairness (Gini): ${this.metrics.fairnessScore?.toFixed(3)}`);
      console.log(`⚠️ Violations: ${this.violations.length}`);
      
      return this.generatedAssignments;
      
    } catch (error) {
      console.error('❌ Critical error in schedule generation:', error);
      this.metrics.totalTime = Date.now() - startTime;
      throw error;
    }
  }
  
  /**
   * Determine optimal call strategy
   */
  private determineCallStrategy(): CallStrategy {
    // Count off-service residents
    const offServiceResidents = this.residents.filter(r => 
      r.service && r.service !== 'Neurosurgery' && r.onService
    );
    
    // Count core neurosurgery residents
    const coreResidents = this.residents.filter(r => 
      (!r.service || r.service === 'Neurosurgery') && r.onService
    );
    
    // Use split strategy if we have enough off-service residents
    // This minimizes post-call burden on core team
    if (this.config.preferSplitCallStrategy && 
        offServiceResidents.length >= 5 && 
        coreResidents.length >= 10) {
      return 'Split';
    }
    
    return 'Standard';
  }
  
  /**
   * Assign calls for a specific day
   */
  private async assignCallsForDay(date: Date): Promise<CallAssignment[]> {
    const assignments: CallAssignment[] = [];
    const callTypes = this.determineCallTypesForDay(date);
    
    for (const callType of callTypes) {
      // Get available residents with all constraints checked
      const available = this.getAvailableResidents(date, callType);
      
      if (available.length === 0) {
        // Try with shortage staffing if normal fails
        if (this.staffingLevel === 'Normal') {
          this.staffingLevel = 'Shortage';
          const shortageAvailable = this.getAvailableResidents(date, callType);
          
          if (shortageAvailable.length > 0) {
            const selected = this.selectResident(shortageAvailable, callType, date);
            if (selected) {
              const assignment = this.createAssignment(selected, date, callType);
              assignments.push(assignment);
              this.updateResidentStats(selected.id, assignment);
              
              // Check PGY-1 supervision requirement
              if (selected.pgyLevel === 1 && this.requiresBackup(callType)) {
                const backup = this.assignBackupCall(date, callType, selected.id);
                if (backup) {
                  assignments.push(backup);
                  this.updateResidentStats(backup.residentId, backup);
                }
              }
              continue;
            }
          }
        }
        
        throw new Error(`No residents available for ${callType} on ${date.toDateString()}`);
      }
      
      // Select best resident using scoring algorithm
      const selected = this.selectResident(available, callType, date);
      
      if (selected) {
        const assignment = this.createAssignment(selected, date, callType);
        assignments.push(assignment);
        this.updateResidentStats(selected.id, assignment);
        
        // Check PGY-1 supervision requirement
        if (selected.pgyLevel === 1 && this.requiresBackup(callType)) {
          const backup = this.assignBackupCall(date, callType, selected.id);
          if (backup) {
            assignments.push(backup);
            this.updateResidentStats(backup.residentId, backup);
          }
        }
      }
    }
    
    return assignments;
  }
  
  /**
   * Get available residents for a call with all constraints
   */
  private getAvailableResidents(date: Date, callType: CallType): Resident[] {
    const dateRotationBlock = this.getRotationBlockForDate(date);
    
    return this.residents.filter(resident => {
      // HARD CONSTRAINT: Vacation & Leave Protection
      if (this.isOnLeave(resident.id, date)) {
        return false;
      }
      
      // HARD CONSTRAINT: Post-Call Protection
      // Day calls do NOT grant post-call (key rule for clinical availability)
      if (this.isPostCall(resident.id, date)) {
        return false;
      }
      
      // Check if already assigned this day
      if (this.hasCallOnDate(resident.id, date)) {
        return false;
      }
      
      const stats = this.statsCache.get(resident.id) || this.callStats.get(resident.id);
      if (!stats) return false;
      
      const workingDays = this.residentWorkingDays.get(resident.id) || 0;
      if (workingDays <= 0) return false;
      
      // HARD CONSTRAINT: PARO Hard Cap
      const paroCap = this.getPAROCap(workingDays);
      if (stats.totalCalls >= paroCap) {
        return false;
      }
      
      // HARD CONSTRAINT: Maximum 2 Weekends per rotation block
      if (callType === 'Weekend') {
        const weekendBlock = this.getWeekendBlock(date);
        
        // Check if already assigned to this weekend
        if (weekendBlock && stats.weekendBlocks.has(weekendBlock)) {
          return false;
        }
        
        // Count weekends in current rotation block only
        let weekendsInRotation = 0;
        stats.weekendBlocks.forEach(block => {
          const blockDate = this.parseWeekendBlock(block);
          if (blockDate) {
            const blockRotation = this.getRotationBlockForDate(blockDate);
            if (blockRotation && dateRotationBlock && 
                blockRotation.blockId === dateRotationBlock.blockId) {
              weekendsInRotation++;
            }
          }
        });
        
        if (weekendsInRotation >= 2) {
          return false;
        }
      }
      
      // INTELLIGENT ASSIGNMENT: Two-Tiered Max Call Target
      if (this.staffingLevel === 'Normal') {
        const pgyTarget = this.getPGYTarget(resident.pgyLevel, workingDays);
        if (stats.totalCalls >= pgyTarget) {
          return false;
        }
      }
      // In shortage mode, only PARO cap applies (already checked above)
      
      // Additional soft constraints
      if (stats.consecutiveDays >= 2) {
        return false;
      }
      
      if (stats.lastCallDate) {
        const daysSince = this.getDaysBetween(date, stats.lastCallDate);
        const minSpacing = this.staffingLevel === 'Shortage' ? 1 : 2;
        if (daysSince < minSpacing) {
          return false;
        }
      }
      
      return true;
    });
  }
  
  /**
   * Select best resident using scoring algorithm
   */
  private selectResident(available: Resident[], callType: CallType, date: Date): Resident | null {
    if (!available || available.length === 0) return null;
    
    const scored = available.map(resident => {
      const stats = this.statsCache.get(resident.id) || this.callStats.get(resident.id);
      if (!stats) return { resident, score: 0 };
      
      let score = 0;
      const weights = this.config.weights;
      
      // CALL EQUALIZATION: Heavily prioritize those with fewer calls
      const avgCalls = this.getAverageCallCount();
      const callDifference = avgCalls - stats.totalCalls;
      score += callDifference * weights.callEqualization;
      
      // SENIORITY: Lower priority for senior residents on primary calls
      if (callType !== 'Backup') {
        score += resident.pgyLevel * weights.seniority;
      } else {
        // For backup, prefer senior residents
        score -= resident.pgyLevel * weights.seniority;
      }
      
      // CHIEF EXEMPTION: Extreme penalty for chiefs
      if (resident.isChief || resident.callExempt) {
        score += weights.chiefExemption; // -1000 by default
      }
      
      // Weekend distribution
      if (callType === 'Weekend') {
        const dateRotationBlock = this.getRotationBlockForDate(date);
        let weekendsInRotation = 0;
        
        stats.weekendBlocks.forEach(block => {
          const blockDate = this.parseWeekendBlock(block);
          if (blockDate) {
            const blockRotation = this.getRotationBlockForDate(blockDate);
            if (blockRotation && dateRotationBlock && 
                blockRotation.blockId === dateRotationBlock.blockId) {
              weekendsInRotation++;
            }
          }
        });
        
        score += (2 - weekendsInRotation) * weights.weekendDistribution;
      }
      
      // Night call preference for off-service (split strategy)
      if (callType === 'Night' && this.callStrategy === 'Split') {
        if (resident.service && resident.service !== 'Neurosurgery') {
          score += weights.offServiceNight;
        }
      }
      
      // Spacing bonus
      if (stats.lastCallDate) {
        const daysSince = this.getDaysBetween(date, stats.lastCallDate);
        score += Math.min(10, daysSince) * weights.spacing;
      } else {
        score += 30; // Never been on call
      }
      
      // Call type variety bonus
      if (!stats.callTypeHistory.includes(callType)) {
        score += weights.variety;
      }
      
      return { resident, score };
    });
    
    // Sort by score (highest first)
    scored.sort((a, b) => b.score - a.score);
    
    return scored[0]?.resident || null;
  }
  
  /**
   * Assign backup call for PGY-1 supervision
   */
  private assignBackupCall(date: Date, primaryCallType: CallType, pgy1Id: string): CallAssignment | null {
    // Determine minimum PGY level for backup
    const minPGY = this.config.minimumBackupSeniority[
      primaryCallType === 'Day' ? 'day' : 
      (primaryCallType === 'Night' || primaryCallType === 'Weekend') ? 'night' : 
      'day'
    ];
    
    const availableBackups = this.residents.filter(resident => {
      if (resident.id === pgy1Id) return false;
      if (resident.pgyLevel < minPGY) return false;
      if (this.isOnLeave(resident.id, date)) return false;
      if (this.hasCallOnDate(resident.id, date)) return false;
      
      // Backup calls can exceed normal limits if needed
      const stats = this.statsCache.get(resident.id);
      if (!stats) return false;
      
      const workingDays = this.residentWorkingDays.get(resident.id) || 0;
      const paroCap = this.getPAROCap(workingDays);
      
      // Still respect PARO hard cap
      return stats.totalCalls < paroCap;
    });
    
    if (availableBackups.length === 0) {
      this.violations.push({
        type: 'hard',
        rule: 'PGY-1 Supervision',
        residentId: pgy1Id,
        date,
        description: `No backup available for PGY-1 on ${date.toDateString()}`
      });
      return null;
    }
    
    const backup = this.selectResident(availableBackups, 'Backup', date);
    if (backup) {
      return this.createAssignment(backup, date, 'Backup');
    }
    
    return null;
  }
  
  /**
   * AI Optimization using simulated annealing
   */
  private async optimizeSchedule(): Promise<OptimizationResult> {
    const startTime = Date.now();
    let currentSchedule = [...this.generatedAssignments];
    let currentScore = this.evaluateSchedule(currentSchedule);
    let bestSchedule = [...currentSchedule];
    let bestScore = currentScore;
    
    // Simulated annealing parameters
    let temperature = 100;
    const coolingRate = 0.995;
    const minTemperature = 0.01;
    
    let iterations = 0;
    let validSwaps = 0;
    let invalidSwaps = 0;
    const maxIterations = this.config.optimizationIterations;
    
    while (temperature > minTemperature && iterations < maxIterations) {
      // Generate neighbor with validation
      const swapResult = this.generateValidNeighbor(currentSchedule);
      
      if (swapResult.isValid) {
        validSwaps++;
        const newScore = this.evaluateSchedule(swapResult.schedule);
        const delta = newScore - currentScore;
        
        // Accept better solutions or worse with probability
        if (delta > 0 || Math.random() < Math.exp(delta / temperature)) {
          currentSchedule = swapResult.schedule;
          currentScore = newScore;
          
          if (currentScore > bestScore) {
            bestSchedule = [...currentSchedule];
            bestScore = currentScore;
          }
        }
      } else {
        invalidSwaps++;
      }
      
      temperature *= coolingRate;
      iterations++;
      
      if (iterations % 1000 === 0) {
        console.log(`  Iteration ${iterations}: Score = ${bestScore.toFixed(3)}`);
      }
    }
    
    const duration = (Date.now() - startTime) / 1000;
    const improvement = bestScore - this.evaluateSchedule(this.generatedAssignments);
    
    console.log(`✅ Optimization complete in ${duration.toFixed(2)}s`);
    console.log(`📊 Score improvement: ${improvement.toFixed(3)}`);
    console.log(`🔄 Valid swaps: ${validSwaps}, Invalid: ${invalidSwaps}`);
    
    return {
      schedule: bestSchedule,
      metrics: this.calculateMetrics(bestSchedule),
      improvements: {
        scoreImprovement: improvement,
        giniImprovement: 0, // Calculate if needed
        violationReduction: { hard: 0, soft: 0 },
        swapsMade: validSwaps
      },
      iterations,
      duration
    };
  }
  
  /**
   * Generate neighbor schedule with constraint validation
   */
  private generateValidNeighbor(schedule: CallAssignment[]): {
    schedule: CallAssignment[];
    isValid: boolean;
  } {
    const newSchedule = [...schedule];
    
    // Find swappable assignments (not PostCall or Backup)
    const swappable = newSchedule.filter(a => 
      a.type !== 'PostCall' && a.type !== 'Backup'
    );
    
    if (swappable.length < 2) {
      return { schedule: newSchedule, isValid: false };
    }
    
    // Select two random assignments
    const idx1 = Math.floor(Math.random() * swappable.length);
    let idx2 = Math.floor(Math.random() * swappable.length);
    while (idx2 === idx1) {
      idx2 = Math.floor(Math.random() * swappable.length);
    }
    
    const assignment1 = swappable[idx1];
    const assignment2 = swappable[idx2];
    
    // Validate swap
    const resident1 = this.residents.find(r => r.id === assignment1.residentId);
    const resident2 = this.residents.find(r => r.id === assignment2.residentId);
    
    if (!resident1 || !resident2) {
      return { schedule: newSchedule, isValid: false };
    }
    
    // Check if swap violates constraints
    if (!this.isSwapValid(resident1, assignment2.date.toDate(), assignment2.type as CallType) ||
        !this.isSwapValid(resident2, assignment1.date.toDate(), assignment1.type as CallType)) {
      return { schedule: newSchedule, isValid: false };
    }
    
    // Perform swap
    const tempId = assignment1.residentId;
    const tempName = assignment1.residentName;
    assignment1.residentId = assignment2.residentId;
    assignment1.residentName = assignment2.residentName;
    assignment2.residentId = tempId;
    assignment2.residentName = tempName;
    
    return { schedule: newSchedule, isValid: true };
  }
  
  /**
   * Check if a swap is valid
   */
  private isSwapValid(resident: Resident, date: Date, callType: CallType): boolean {
    // Check vacation/leave
    if (this.isOnLeave(resident.id, date)) return false;
    
    // Check post-call
    if (this.isPostCall(resident.id, date)) return false;
    
    // Check PARO cap
    const stats = this.callStats.get(resident.id);
    if (!stats) return false;
    
    const workingDays = this.residentWorkingDays.get(resident.id) || 0;
    const paroCap = this.getPAROCap(workingDays);
    
    if (stats.totalCalls >= paroCap) return false;
    
    // Check weekend limits
    if (callType === 'Weekend') {
      const weekendBlock = this.getWeekendBlock(date);
      if (!weekendBlock) return false;
      
      if (stats.weekendBlocks.has(weekendBlock)) return false;
      
      // Check rotation block limit
      const dateRotationBlock = this.getRotationBlockForDate(date);
      let weekendsInRotation = 0;
      
      stats.weekendBlocks.forEach(block => {
        const blockDate = this.parseWeekendBlock(block);
        if (blockDate) {
          const blockRotation = this.getRotationBlockForDate(blockDate);
          if (blockRotation && dateRotationBlock && 
              blockRotation.blockId === dateRotationBlock.blockId) {
            weekendsInRotation++;
          }
        }
      });
      
      if (weekendsInRotation >= 2) return false;
    }
    
    // Check PGY-1 supervision
    if (resident.pgyLevel === 1 && this.requiresBackup(callType)) {
      const backupAvailable = this.residents.some(r => 
        r.id !== resident.id &&
        r.pgyLevel >= 2 &&
        !this.isOnLeave(r.id, date) &&
        !this.hasCallOnDate(r.id, date)
      );
      
      if (!backupAvailable) return false;
    }
    
    return true;
  }
  
  /**
   * Evaluate schedule quality
   */
  private evaluateSchedule(schedule: CallAssignment[]): number {
    // Calculate metrics
    const metrics = this.calculateMetrics(schedule);
    
    // Base score from fairness (inverse of Gini)
    let score = 1 - Math.abs(metrics.giniCoefficient);
    
    // Penalize violations
    score -= metrics.violations.hard * 0.5;
    score -= metrics.violations.soft * 0.01;
    
    // Bonus for even distribution
    const variance = metrics.distributionVariance || 0;
    score += (1 - Math.min(1, variance)) * 0.2;
    
    return Math.max(0, score);
  }
  
  /**
   * Calculate comprehensive metrics
   */
  private calculateMetrics(schedule: CallAssignment[]): ScheduleAnalytics {
    const residentCalls = new Map<string, number>();
    const residentPoints = new Map<string, number>();
    
    // Count calls and points
    schedule.forEach(assignment => {
      if (assignment.type === 'PostCall') return;
      
      const currentCalls = residentCalls.get(assignment.residentId) || 0;
      const currentPoints = residentPoints.get(assignment.residentId) || 0;
      
      residentCalls.set(assignment.residentId, currentCalls + 1);
      residentPoints.set(assignment.residentId, currentPoints + (assignment.points || 1));
    });
    
    // Calculate Gini coefficient
    const callCounts = Array.from(residentCalls.values()).sort((a, b) => a - b);
    const gini = this.calculateGiniCoefficient(callCounts);
    
    // Calculate variance
    const mean = callCounts.reduce((a, b) => a + b, 0) / Math.max(1, callCounts.length);
    const variance = callCounts.reduce((sum, count) => 
      sum + Math.pow(count - mean, 2), 0
    ) / Math.max(1, callCounts.length);
    
    // Count violations
    const violations = this.countViolations(schedule);
    
    return {
      totalCalls: schedule.filter(a => a.type !== 'PostCall').length,
      residentDistribution: Object.fromEntries(residentCalls),
      pointDistribution: Object.fromEntries(residentPoints),
      giniCoefficient: gini,
      distributionVariance: variance,
      standardDeviation: Math.sqrt(variance),
      violations,
      weekendCoverage: this.calculateWeekendCoverage(schedule),
      pgy1Supervision: this.validatePGY1Supervision(schedule)
    };
  }
  
  /**
   * Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality)
   */
  private calculateGiniCoefficient(values: number[]): number {
    if (values.length === 0) return 0;
    
    const n = values.length;
    const sorted = [...values].sort((a, b) => a - b);
    
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += (2 * (i + 1) - n - 1) * sorted[i];
    }
    
    const total = sorted.reduce((a, b) => a + b, 0);
    if (total === 0) return 0;
    
    return sum / (n * total);
  }
  
  /**
   * Generate comprehensive analytics
   */
  public generateAnalytics(): {
    overview: any;
    distribution: any[];
    violations: any;
    fairness: any;
  } {
    const overview = {
      month: `${this.month + 1}/${this.year}`,
      totalAssignments: this.generatedAssignments.length,
      totalResidents: this.residents.length,
      staffingLevel: this.staffingLevel,
      callStrategy: this.callStrategy,
      generationTime: `${this.metrics.totalTime}ms`,
      optimizationImprovement: this.metrics.optimizationImprovement 
        ? `${this.metrics.optimizationImprovement.toFixed(3)}` 
        : 'Not optimized'
    };
    
    const distribution = this.residents.map(resident => {
      const stats = this.callStats.get(resident.id);
      const workingDays = this.residentWorkingDays.get(resident.id) || 0;
      
      return {
        name: resident.name,
        pgyLevel: resident.pgyLevel,
        totalCalls: stats?.totalCalls || 0,
        weekendCalls: stats?.weekendCalls || 0,
        holidayCalls: stats?.holidayCalls || 0,
        nightCalls: stats?.nightCalls || 0,
        dayCalls: stats?.dayCalls || 0,
        backupCalls: stats?.backupCalls || 0,
        points: stats?.points || 0,
        workingDays,
        callRatio: workingDays > 0 && stats?.totalCalls 
          ? `1-in-${Math.round(workingDays / stats.totalCalls)}`
          : 'N/A',
        isChief: resident.isChief || false,
        callExempt: resident.callExempt || false
      };
    });
    
    const violations = {
      total: this.violations.length,
      hard: this.violations.filter(v => v.type === 'hard').length,
      soft: this.violations.filter(v => v.type === 'soft').length,
      details: this.violations
    };
    
    const fairness = {
      giniCoefficient: this.metrics.fairnessScore?.toFixed(3) || 'N/A',
      interpretation: this.interpretGiniScore(this.metrics.fairnessScore || 0),
      averageCallsPerResident: (
        this.generatedAssignments.filter(a => a.type !== 'PostCall').length / 
        Math.max(1, this.residents.length)
      ).toFixed(2),
      distributionChart: this.generateDistributionChart()
    };
    
    return {
      overview,
      distribution,
      violations,
      fairness
    };
  }
  
  // === Helper Methods (Keeping existing functionality) ===
  
  private validateInputs(residents: Resident[], month: number, year: number): void {
    if (!residents || !Array.isArray(residents)) {
      throw new Error('Residents must be a valid array');
    }
    if (residents.length === 0) {
      throw new Error('At least one resident is required');
    }
    if (month < 0 || month > 11 || !Number.isInteger(month)) {
      throw new Error(`Invalid month: ${month}`);
    }
    if (year < 2020 || year > 2050 || !Number.isInteger(year)) {
      throw new Error(`Invalid year: ${year}`);
    }
  }
  
  private initializeCallStats(): void {
    this.callStats.clear();
    this.statsCache.clear();
    
    this.residents.forEach(resident => {
      const existingCalls = this.existingAssignments.filter(
        a => a.residentId === resident.id && a.type !== 'PostCall'
      );
      
      const stats: CallStats = {
        totalCalls: existingCalls.length,
        weekendCalls: existingCalls.filter(a => a.type === 'Weekend').length,
        holidayCalls: existingCalls.filter(a => a.type === 'Holiday').length,
        nightCalls: existingCalls.filter(a => a.type === 'Night').length,
        dayCalls: existingCalls.filter(a => a.type === 'Day').length,
        twentyFourHourCalls: existingCalls.filter(a => a.type === '24h').length,
        backupCalls: existingCalls.filter(a => a.type === 'Backup').length,
        lastCallDate: this.getLastCallDate(existingCalls),
        consecutiveDays: 0,
        callDates: existingCalls.map(a => a.date.toDate()),
        points: existingCalls.reduce((sum, a) => sum + (a.points || 0), 0),
        weekendBlocks: new Set(),
        callTypeHistory: existingCalls
          .sort((a, b) => b.date.toDate().getTime() - a.date.toDate().getTime())
          .slice(0, 5)
          .map(a => a.type as CallType),
        workingDaysInRotation: 0,
        giniContribution: 0,
        currentRotationBlock: this.currentRotationBlock?.blockId || ''
      };
      
      // Track weekend blocks
      existingCalls.forEach(assignment => {
        if (assignment.type === 'Weekend') {
          const weekendBlock = this.getWeekendBlock(assignment.date.toDate());
          if (weekendBlock) {
            stats.weekendBlocks.add(weekendBlock);
          }
        }
      });
      
      this.callStats.set(resident.id, stats);
      this.statsCache.set(resident.id, { ...stats });
    });
  }
  
  private precalculateCalendarInfo(): void {
    const daysInMonth = new Date(this.year, this.month + 1, 0).getDate();
    
    for (let day = 1; day <= daysInMonth; day++) {
      const date = new Date(this.year, this.month, day);
      const dayOfWeek = date.getDay();
      
      if (dayOfWeek === 0 || dayOfWeek === 6) {
        this.weekends.push(date);
      }
      
      if (this.isHoliday(date)) {
        this.holidays.push(date);
      }
    }
    
    console.log(`📅 Month has ${daysInMonth} days, ${this.weekends.length} weekend days, ${this.holidays.length} holidays`);
  }
  
  private calculateWorkingDays(): void {
    const daysInMonth = new Date(this.year, this.month + 1, 0).getDate();
    
    this.residents.forEach(resident => {
      const residentLeave = this.approvedLeave.filter(
        leave => leave.residentId === resident.id && leave.status === 'Approved'
      );
      
      const leaveDaySet = new Set<string>();
      
      residentLeave.forEach(leave => {
        const startDate = leave.startDate.toDate();
        const endDate = leave.endDate.toDate();
        
        for (let d = new Date(startDate); d <= endDate; d.setDate(d.getDate() + 1)) {
          if (d.getMonth() === this.month && d.getFullYear() === this.year) {
            leaveDaySet.add(d.toDateString());
          }
        }
      });
      
      const workingDays = Math.max(1, daysInMonth - leaveDaySet.size);
      this.residentWorkingDays.set(resident.id, workingDays);
      
      // Update stats
      const stats = this.callStats.get(resident.id);
      if (stats) {
        stats.workingDaysInRotation = workingDays;
        this.statsCache.set(resident.id, { ...stats });
      }
    });
  }
  
  private identifyWeekendBlocks(): void {
    const checkStart = new Date(this.year, this.month, -2);
    const checkEnd = new Date(this.year, this.month + 1, 3);
    
    for (let d = new Date(checkStart); d <= checkEnd; d.setDate(d.getDate() + 1)) {
      const dayOfWeek = d.getDay();
      
      if (dayOfWeek === 5 || dayOfWeek === 6 || dayOfWeek === 0) {
        const block = this.getWeekendBlock(d);
        if (block) {
          this.weekendBlockCache.set(d.toDateString(), block);
        }
      }
    }
  }
  
  private processPreviousMonthPostCalls(previousPostCalls: CrossMonthPostCall[]): void {
    const firstOfMonth = new Date(this.year, this.month, 1);
    
    for (const postCall of previousPostCalls) {
      const postCallDate = postCall.date instanceof Date ? 
        postCall.date : postCall.date.toDate();
      
      if (this.isSameDay(postCallDate, firstOfMonth)) {
        const resident = this.residents.find(r => r.id === postCall.residentId);
        
        if (resident) {
          const assignment = this.createAssignment(resident, firstOfMonth, 'PostCall');
          
          if (postCall.isPostHoliday) {
            assignment.isHolidayPostCall = true;
          }
          
          this.generatedAssignments.push(assignment);
          console.log(`✅ Created post-call for ${resident.name} on ${firstOfMonth.toDateString()}`);
        }
      }
    }
  }
  
  private determineCallTypesForDay(date: Date): CallType[] {
    const dayOfWeek = date.getDay();
    const isWeekend = dayOfWeek === 0 || dayOfWeek === 6 || dayOfWeek === 5;
    const isHoliday = this.isHoliday(date);
    
    if (isHoliday) {
      return ['Holiday'];
    }
    
    if (isWeekend) {
      return ['Weekend'];
    }
    
    // Weekday: Use strategy
    if (this.callStrategy === 'Split') {
      return ['Day', 'Night'];
    }
    
    return ['24h'];
  }
  
  private requiresBackup(callType: CallType): boolean {
    return callType !== 'Backup' && callType !== 'PostCall';
  }
  
  private createAssignment(resident: Resident, date: Date, callType: CallType): CallAssignment {
    const points = this.calculatePoints(callType);
    
    return {
      id: `call-${uuidv4()}`,
      residentId: resident.id,
      residentName: resident.name,
      date: admin.firestore.Timestamp.fromDate(date),
      type: callType,
      points: points,
      status: 'Scheduled',
      createdAt: admin.firestore.Timestamp.now(),
      updatedAt: admin.firestore.Timestamp.now(),
      isEmergencyAssignment: false
    };
  }
  
  private calculatePoints(callType: CallType): number {
    switch (callType) {
      case 'Holiday':
        return 3;
      case '24h':
        return 3;
      case 'Weekend':
        return 2;
      case 'Night':
        return 1.5;
      case 'Day':
      case 'Backup':
        return 1;
      case 'PostCall':
        return 0;
      default:
        return 1;
    }
  }
  
  private updateResidentStats(residentId: string, assignment: CallAssignment): void {
    const stats = this.callStats.get(residentId);
    if (!stats) return;
    
    stats.totalCalls++;
    stats.lastCallDate = assignment.date.toDate();
    stats.points += assignment.points;
    stats.callDates.push(assignment.date.toDate());
    
    switch (assignment.type) {
      case 'Weekend':
        stats.weekendCalls++;
        const weekendBlock = this.getWeekendBlock(assignment.date.toDate());
        if (weekendBlock) {
          stats.weekendBlocks.add(weekendBlock);
        }
        break;
      case 'Holiday':
        stats.holidayCalls++;
        break;
      case 'Night':
        stats.nightCalls++;
        break;
      case 'Day':
        stats.dayCalls++;
        break;
      case '24h':
        stats.twentyFourHourCalls++;
        break;
      case 'Backup':
        stats.backupCalls++;
        break;
    }
    
    stats.callTypeHistory.unshift(assignment.type as CallType);
    stats.callTypeHistory = stats.callTypeHistory.slice(0, 5);
    
    // Sync to cache
    const statsCopy = {
      ...stats,
      weekendBlocks: new Set(stats.weekendBlocks)
    };
    this.statsCache.set(residentId, statsCopy);
  }
  
  private updateCallStatsForDay(assignments: CallAssignment[]): void {
    // Reset consecutive days
    this.callStats.forEach((stats, residentId) => {
      stats.consecutiveDays = 0;
      const cachedStats = this.statsCache.get(residentId);
      if (cachedStats) {
        cachedStats.consecutiveDays = 0;
      }
    });
    
    // Update for assigned residents
    assignments.forEach(assignment => {
      if (assignment.type !== 'PostCall') {
        const stats = this.callStats.get(assignment.residentId);
        const cachedStats = this.statsCache.get(assignment.residentId);
        if (stats) {
          stats.consecutiveDays = 1;
        }
        if (cachedStats) {
          cachedStats.consecutiveDays = 1;
        }
      }
    });
  }
  
  private syncStatsCache(): void {
    this.callStats.forEach((stats, residentId) => {
      const statsCopy = {
        ...stats,
        weekendBlocks: new Set(stats.weekendBlocks)
      };
      this.statsCache.set(residentId, statsCopy);
    });
  }
  
  private generatePostCallAssignments(): CallAssignment[] {
    const postCallAssignments: CallAssignment[] = [];
    
    for (const assignment of this.generatedAssignments) {
      // Only certain call types grant post-call protection
      // Day calls do NOT grant post-call
      if (assignment.type === '24h' || assignment.type === 'Weekend' || 
          assignment.type === 'Night' || assignment.type === 'Holiday') {
        
        const postCallDate = new Date(assignment.date.toDate());
        postCallDate.setDate(postCallDate.getDate() + 1);
        
        // Check if crosses month boundary
        if (postCallDate.getMonth() !== this.month || 
            postCallDate.getFullYear() !== this.year) {
          
          this.crossMonthPostCalls.push({
            residentId: assignment.residentId,
            date: admin.firestore.Timestamp.fromDate(postCallDate),
            originatingCallId: assignment.id,
            originatingCallDate: assignment.date,
            isPostHoliday: assignment.type === 'Holiday'
          });
          continue;
        }
        
        const resident = this.residents.find(r => r.id === assignment.residentId);
        if (resident) {
          const postCallAssignment = this.createAssignment(resident, postCallDate, 'PostCall');
          postCallAssignments.push(postCallAssignment);
        }
      }
    }
    
    return postCallAssignments;
  }
  
  private calculateFairnessMetrics(): void {
    const callCounts: number[] = [];
    
    this.callStats.forEach(stats => {
      callCounts.push(stats.totalCalls);
    });
    
    if (callCounts.length === 0) {
      this.metrics.fairnessScore = 0;
      return;
    }
    
    this.metrics.fairnessScore = this.calculateGiniCoefficient(callCounts);
    this.metrics.violationCount = this.violations.length;
  }
  
  private isOnLeave(residentId: string, date: Date): boolean {
    return this.approvedLeave.some(leave => {
      if (leave.residentId !== residentId) return false;
      if (leave.status !== 'Approved') return false;
      
      const startDate = leave.startDate.toDate();
      const endDate = leave.endDate.toDate();
      
      return date >= startDate && date <= endDate;
    });
  }
  
  private isPostCall(residentId: string, date: Date): boolean {
    const previousDay = new Date(date);
    previousDay.setDate(date.getDate() - 1);
    
    // Check generated assignments
    const hadProtectedCall = this.generatedAssignments.some(a => {
      if (a.residentId !== residentId) return false;
      if (!this.isSameDay(a.date.toDate(), previousDay)) return false;
      
      // Day calls do NOT grant post-call protection
      return a.type === '24h' || a.type === 'Weekend' || 
             a.type === 'Night' || a.type === 'Holiday';
    });
    
    if (!hadProtectedCall) {
      // Check existing assignments
      return this.existingAssignments.some(a => {
        if (a.residentId !== residentId) return false;
        if (!this.isSameDay(a.date.toDate(), previousDay)) return false;
        return a.type === '24h' || a.type === 'Weekend' || 
               a.type === 'Night' || a.type === 'Holiday';
      });
    }
    
    return hadProtectedCall;
  }
  
  private hasCallOnDate(residentId: string, date: Date): boolean {
    const dateStr = date.toDateString();
    
    const hasExisting = this.existingAssignments.some(a => 
      a.residentId === residentId && 
      a.date.toDate().toDateString() === dateStr
    );
    
    if (hasExisting) return true;
    
    return this.generatedAssignments.some(a =>
      a.residentId === residentId &&
      a.date.toDate().toDateString() === dateStr
    );
  }
  
  private getPAROCap(workingDays: number): number {
    const rule = this.config.paroHardCaps.find(
      cap => workingDays >= cap.minDays && workingDays <= cap.maxDays
    );
    return rule ? rule.maxCalls : 7;
  }
  
  private getPGYTarget(pgyLevel: number, workingDays: number): number {
    const ratio = this.config.pgyCallRatios[pgyLevel] || 7;
    return Math.floor(workingDays / ratio);
  }
  
  private getAverageCallCount(): number {
    let totalCalls = 0;
    let residentCount = 0;
    
    this.callStats.forEach(stats => {
      totalCalls += stats.totalCalls;
      residentCount++;
    });
    
    return residentCount > 0 ? totalCalls / residentCount : 0;
  }
  
  private getDaysBetween(date1: Date, date2: Date): number {
    const timeDiff = Math.abs(date1.getTime() - date2.getTime());
    return Math.floor(timeDiff / (1000 * 60 * 60 * 24));
  }
  
  private isSameDay(date1: Date, date2: Date): boolean {
    return date1.getFullYear() === date2.getFullYear() && 
           date1.getMonth() === date2.getMonth() &&
           date1.getDate() === date2.getDate();
  }
  
  private isHoliday(date: Date): boolean {
    // Check predefined holidays
    if (this.holidays.some(holiday => this.isSameDay(date, holiday))) {
      return true;
    }
    
    // Check academic year holidays
    if (this.academicYear.holidays.some(holiday => {
      const holidayDate = holiday.date.toDate();
      return this.isSameDay(date, holidayDate);
    })) {
      return true;
    }
    
    // Check statutory holidays (simplified)
    const month = date.getMonth();
    const day = date.getDate();
    
    // New Year's Day
    if (month === 0 && day === 1) return true;
    
    // Canada Day
    if (month === 6 && day === 1) return true;
    
    // Christmas
    if (month === 11 && day === 25) return true;
    
    // Boxing Day
    if (month === 11 && day === 26) return true;
    
    return false;
  }
  
  private getWeekendBlock(date: Date): string | null {
    const dayOfWeek = date.getDay();
    
    if (dayOfWeek !== 0 && dayOfWeek !== 5 && dayOfWeek !== 6) {
      return null;
    }
    
    // Find Friday of this weekend
    const friday = new Date(date);
    
    if (dayOfWeek === 5) {
      // Already Friday
    } else if (dayOfWeek === 6) {
      friday.setDate(date.getDate() - 1);
    } else if (dayOfWeek === 0) {
      friday.setDate(date.getDate() - 2);
    }
    
    // Handle cross-month weekends
    if (friday.getMonth() !== date.getMonth()) {
      const saturday = new Date(date);
      if (dayOfWeek === 5) {
        saturday.setDate(date.getDate() + 1);
      } else if (dayOfWeek === 0) {
        saturday.setDate(date.getDate() - 1);
      }
      return `${saturday.getFullYear()}-${saturday.getMonth()}-${saturday.getDate()}-crossmonth`;
    }
    
    return `${friday.getFullYear()}-${friday.getMonth()}-${friday.getDate()}`;
  }
  
  private parseWeekendBlock(block: string): Date | null {
    const parts = block.split('-');
    if (parts.length >= 3) {
      return new Date(
        parseInt(parts[0]),
        parseInt(parts[1]),
        parseInt(parts[2])
      );
    }
    return null;
  }
  
  private getRotationBlockForDate(date: Date): RotationBlockInfo | null {
    for (const [_, block] of this.rotationBlocks) {
      if (date >= block.startDate && date <= block.endDate) {
        return block;
      }
    }
    return null;
  }
  
  private getLastCallDate(assignments: CallAssignment[]): Date | null {
    if (!assignments || assignments.length === 0) return null;
    
    const sorted = assignments.sort((a, b) => 
      b.date.toDate().getTime() - a.date.toDate().getTime()
    );
    
    return sorted[0].date.toDate();
  }
  
  private countViolations(schedule: CallAssignment[]): { hard: number; soft: number } {
    let hardViolations = 0;
    let softViolations = 0;
    
    // Count from tracked violations
    this.violations.forEach(v => {
      if (v.type === 'hard') hardViolations++;
      else softViolations++;
    });
    
    return { hard: hardViolations, soft: softViolations };
  }
  
  private calculateWeekendCoverage(schedule: CallAssignment[  private calculateWeekendCoverage(schedule: CallAssignment[]): number {
    const weekendCalls = schedule.filter(a => a.type === 'Weekend').length;
    const totalWeekends = Math.floor(this.weekends.length / 3); // 3 days per weekend
    
    return totalWeekends > 0 ? (weekendCalls / totalWeekends) * 100 : 100;
  }
  
  private validatePGY1Supervision(schedule: CallAssignment[]): boolean {
    const pgy1Calls = schedule.filter(a => {
      const resident = this.residents.find(r => r.id === a.residentId);
      return resident?.pgyLevel === 1 && a.type !== 'PostCall' && a.type !== 'Backup';
    });
    
    for (const call of pgy1Calls) {
      const hasBackup = schedule.some(a => 
        a.type === 'Backup' &&
        this.isSameDay(a.date.toDate(), call.date.toDate())
      );
      
      if (!hasBackup) return false;
    }
    
    return true;
  }
  
  private interpretGiniScore(gini: number): string {
    if (gini < 0.1) return 'Excellent - Very fair distribution';
    if (gini < 0.2) return 'Good - Fair distribution';
    if (gini < 0.3) return 'Moderate - Some inequality';
    if (gini < 0.4) return 'Poor - Significant inequality';
    return 'Very Poor - Severe inequality';
  }
  
  private generateDistributionChart(): string[] {
    const chart: string[] = [];
    const maxCalls = Math.max(...Array.from(this.callStats.values()).map(s => s.totalCalls));
    
    this.residents
      .sort((a, b) => a.pgyLevel - b.pgyLevel)
      .forEach(resident => {
        const stats = this.callStats.get(resident.id);
        const calls = stats?.totalCalls || 0;
        const barLength = Math.round((calls / Math.max(1, maxCalls)) * 20);
        const bar = '█'.repeat(barLength) + '░'.repeat(20 - barLength);
        
        const chiefIndicator = resident.isChief ? ' [CHIEF]' : '';
        const exemptIndicator = resident.callExempt ? ' [EXEMPT]' : '';
        
        chart.push(`PGY-${resident.pgyLevel} ${resident.name.padEnd(20)} ${bar} ${calls}${chiefIndicator}${exemptIndicator}`);
      });
    
    return chart;
  }
  
  /**
   * Get cross-month post calls for next month
   */
  public getCrossMonthPostCalls(): CrossMonthPostCall[] {
    return this.crossMonthPostCalls;
  }
  
  /**
   * Get all violations
   */
  public getViolations(): ScheduleViolation[] {
    return this.violations;
  }
  
  /**
   * Get performance metrics
   */
  public getMetrics(): PerformanceMetrics {
    return this.metrics;
  }
  
  /**
   * Validate the complete schedule
   */
  public validateSchedule(): {
    isValid: boolean;
    violations: ScheduleViolation[];
    warnings: string[];
  } {
    const violations: ScheduleViolation[] = [];
    const warnings: string[] = [];
    
    // Track stats for validation
    const validationStats = new Map<string, {
      calls: number;
      weekendBlocks: Set<string>;
      lastCallDate: Date | null;
      postCallDates: Set<string>;
    }>();
    
    // Initialize validation stats
    this.residents.forEach(resident => {
      validationStats.set(resident.id, {
        calls: 0,
        weekendBlocks: new Set(),
        lastCallDate: null,
        postCallDates: new Set()
      });
    });
    
    // Check each assignment
    this.generatedAssignments
      .sort((a, b) => a.date.toDate().getTime() - b.date.toDate().getTime())
      .forEach(assignment => {
        const resident = this.residents.find(r => r.id === assignment.residentId);
        if (!resident) return;
        
        const stats = validationStats.get(resident.id);
        if (!stats) return;
        
        const date = assignment.date.toDate();
        
        // HARD CONSTRAINT: Vacation protection
        if (this.isOnLeave(resident.id, date)) {
          violations.push({
            type: 'hard',
            rule: 'Vacation Protection',
            residentId: resident.id,
            date,
            description: `${resident.name} assigned while on leave`
          });
        }
        
        // HARD CONSTRAINT: Post-call protection
        if (assignment.type !== 'PostCall' && stats.postCallDates.has(date.toDateString())) {
          violations.push({
            type: 'hard',
            rule: 'Post-Call Protection',
            residentId: resident.id,
            date,
            description: `${resident.name} assigned on post-call day`
          });
        }
        
        // Track assignments
        if (assignment.type !== 'PostCall') {
          stats.calls++;
          
          // HARD CONSTRAINT: PARO cap
          const workingDays = this.residentWorkingDays.get(resident.id) || 0;
          const paroCap = this.getPAROCap(workingDays);
          
          if (stats.calls > paroCap) {
            violations.push({
              type: 'hard',
              rule: 'PARO Hard Cap',
              residentId: resident.id,
              date,
              description: `${resident.name} exceeds PARO cap of ${paroCap} calls (has ${stats.calls})`
            });
          }
          
          // HARD CONSTRAINT: Weekend blocks
          if (assignment.type === 'Weekend') {
            const weekendBlock = this.getWeekendBlock(date);
            if (weekendBlock) {
              if (stats.weekendBlocks.has(weekendBlock)) {
                violations.push({
                  type: 'hard',
                  rule: 'Weekend Block',
                  residentId: resident.id,
                  date,
                  description: `${resident.name} assigned multiple times to same weekend`
                });
              }
              stats.weekendBlocks.add(weekendBlock);
              
              // Check 2-weekend maximum per rotation
              const rotationBlock = this.getRotationBlockForDate(date);
              let weekendsInRotation = 0;
              
              stats.weekendBlocks.forEach(block => {
                const blockDate = this.parseWeekendBlock(block);
                if (blockDate) {
                  const blockRotation = this.getRotationBlockForDate(blockDate);
                  if (blockRotation && rotationBlock &&
                      blockRotation.blockId === rotationBlock.blockId) {
                    weekendsInRotation++;
                  }
                }
              });
              
              if (weekendsInRotation > 2) {
                violations.push({
                  type: 'hard',
                  rule: 'Maximum 2 Weekends',
                  residentId: resident.id,
                  date,
                  description: `${resident.name} exceeds 2 weekends in rotation block (has ${weekendsInRotation})`
                });
              }
            }
          }
          
          // HARD CONSTRAINT: PGY-1 supervision
          if (resident.pgyLevel === 1 && this.requiresBackup(assignment.type as CallType)) {
            const hasBackup = this.generatedAssignments.some(a =>
              a.type === 'Backup' &&
              this.isSameDay(a.date.toDate(), date) &&
              a.residentId !== resident.id
            );
            
            if (!hasBackup) {
              violations.push({
                type: 'hard',
                rule: 'PGY-1 Supervision',
                residentId: resident.id,
                date,
                description: `PGY-1 ${resident.name} has no backup supervision`
              });
            }
          }
          
          // Add post-call date if applicable
          // Day calls do NOT grant post-call protection
          if (assignment.type === '24h' || assignment.type === 'Weekend' ||
              assignment.type === 'Night' || assignment.type === 'Holiday') {
            const postCallDate = new Date(date);
            postCallDate.setDate(date.getDate() + 1);
            stats.postCallDates.add(postCallDate.toDateString());
          }
          
          stats.lastCallDate = date;
        }
      });
    
    // Add warnings for soft constraints
    this.residents.forEach(resident => {
      const stats = validationStats.get(resident.id);
      if (!stats) return;
      
      const workingDays = this.residentWorkingDays.get(resident.id) || 0;
      const pgyTarget = this.getPGYTarget(resident.pgyLevel, workingDays);
      
      // Check PGY target (soft constraint)
      if (this.staffingLevel === 'Normal' && stats.calls > pgyTarget) {
        warnings.push(`${resident.name} (PGY-${resident.pgyLevel}) has ${stats.calls} calls, exceeding target of ${pgyTarget}`);
      }
      
      // Check chief exemption (soft constraint)
      if (resident.isChief && stats.calls > 2) {
        warnings.push(`Chief ${resident.name} has ${stats.calls} calls (should be minimized)`);
      }
      
      // Check call-exempt status
      if (resident.callExempt && stats.calls > 0) {
        warnings.push(`Call-exempt resident ${resident.name} has ${stats.calls} calls`);
      }
    });
    
    return {
      isValid: violations.filter(v => v.type === 'hard').length === 0,
      violations,
      warnings
    };
  }
  
  /**
   * Export schedule to various formats
   */
  public exportSchedule(format: 'json' | 'csv' | 'ical' = 'json'): string {
    switch (format) {
      case 'csv':
        return this.exportToCSV();
      case 'ical':
        return this.exportToICal();
      default:
        return this.exportToJSON();
    }
  }
  
  private exportToJSON(): string {
    return JSON.stringify({
      metadata: {
        month: this.month + 1,
        year: this.year,
        generated: new Date().toISOString(),
        generatedBy: 'MonthlyCallScheduler v2.0.0',
        strategy: this.callStrategy,
        staffingLevel: this.staffingLevel
      },
      assignments: this.generatedAssignments.map(a => ({
        id: a.id,
        residentId: a.residentId,
        residentName: a.residentName,
        date: a.date.toDate().toISOString(),
        type: a.type,
        points: a.points,
        status: a.status
      })),
      crossMonthPostCalls: this.crossMonthPostCalls,
      metrics: this.metrics,
      analytics: this.generateAnalytics()
    }, null, 2);
  }
  
  private exportToCSV(): string {
    const headers = ['Date', 'Day', 'Resident', 'PGY', 'Call Type', 'Points', 'Chief', 'Exempt'];
    const rows = [headers.join(',')];
    
    this.generatedAssignments
      .sort((a, b) => a.date.toDate().getTime() - b.date.toDate().getTime())
      .forEach(assignment => {
        const date = assignment.date.toDate();
        const resident = this.residents.find(r => r.id === assignment.residentId);
        const dayName = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][date.getDay()];
        
        rows.push([
          date.toLocaleDateString(),
          dayName,
          assignment.residentName,
          resident?.pgyLevel || '',
          assignment.type,
          assignment.points.toString(),
          resident?.isChief ? 'Yes' : 'No',
          resident?.callExempt ? 'Yes' : 'No'
        ].join(','));
      });
    
    return rows.join('\n');
  }
  
  private exportToICal(): string {
    const lines = [
      'BEGIN:VCALENDAR',
      'VERSION:2.0',
      'PRODID:-//Monthly Call Scheduler//EN',
      'CALSCALE:GREGORIAN',
      'METHOD:PUBLISH',
      `X-WR-CALNAME:Call Schedule ${this.month + 1}/${this.year}`
    ];
    
    this.generatedAssignments.forEach(assignment => {
      const date = assignment.date.toDate();
      const startDate = this.formatICalDate(date);
      const endDate = this.formatICalDate(new Date(date.getTime() + 86400000));
      
      lines.push(
        'BEGIN:VEVENT',
        `UID:${assignment.id}@callscheduler.com`,
        `DTSTART;VALUE=DATE:${startDate}`,
        `DTEND;VALUE=DATE:${endDate}`,
        `SUMMARY:${assignment.residentName} - ${assignment.type} Call`,
        `DESCRIPTION:${assignment.type} call assignment for ${assignment.residentName} (${assignment.points} points)`,
        `CATEGORIES:${assignment.type}`,
        'STATUS:CONFIRMED',
        'END:VEVENT'
      );
    });
    
    lines.push('END:VCALENDAR');
    return lines.join('\r\n');
  }
  
  private formatICalDate(date: Date): string {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}${month}${day}`;
  }
  
  /**
   * Generate a detailed report for administrators
   */
  public generateDetailedReport(): string {
    const analytics = this.generateAnalytics();
    const validation = this.validateSchedule();
    
    let report = '═'.repeat(60) + '\n';
    report += '  MONTHLY CALL SCHEDULE REPORT\n';
    report += '═'.repeat(60) + '\n\n';
    
    // Overview
    report += '📊 OVERVIEW\n';
    report += '─'.repeat(40) + '\n';
    report += `Month: ${analytics.overview.month}\n`;
    report += `Total Assignments: ${analytics.overview.totalAssignments}\n`;
    report += `Total Residents: ${analytics.overview.totalResidents}\n`;
    report += `Staffing Level: ${analytics.overview.staffingLevel}\n`;
    report += `Call Strategy: ${analytics.overview.callStrategy}\n`;
    report += `Generation Time: ${analytics.overview.generationTime}\n`;
    report += `Optimization: ${analytics.overview.optimizationImprovement}\n\n`;
    
    // Fairness Metrics
    report += '⚖️ FAIRNESS METRICS\n';
    report += '─'.repeat(40) + '\n';
    report += `Gini Coefficient: ${analytics.fairness.giniCoefficient}\n`;
    report += `Interpretation: ${analytics.fairness.interpretation}\n`;
    report += `Average Calls/Resident: ${analytics.fairness.averageCallsPerResident}\n\n`;
    
    // Constraint Compliance
    report += '✅ CONSTRAINT COMPLIANCE\n';
    report += '─'.repeat(40) + '\n';
    report += `Schedule Valid: ${validation.isValid ? 'YES ✅' : 'NO ❌'}\n`;
    report += `Hard Violations: ${analytics.violations.hard}\n`;
    report += `Soft Violations: ${analytics.violations.soft}\n`;
    report += `Warnings: ${validation.warnings.length}\n\n`;
    
    // Hard Constraints Status
    report += '🔒 HARD CONSTRAINTS STATUS\n';
    report += '─'.repeat(40) + '\n';
    
    const hardConstraints = [
      { name: 'Vacation & Leave Protection', violations: 0 },
      { name: 'Post-Call Protection', violations: 0 },
      { name: 'PARO Hard Cap', violations: 0 },
      { name: 'Maximum 2 Weekends/Rotation', violations: 0 },
      { name: 'PGY-1 Supervision', violations: 0 }
    ];
    
    validation.violations.forEach(v => {
      if (v.type === 'hard') {
        const constraint = hardConstraints.find(c => c.name.includes(v.rule));
        if (constraint) constraint.violations++;
      }
    });
    
    hardConstraints.forEach(c => {
      const status = c.violations === 0 ? '✅ PASS' : `❌ FAIL (${c.violations} violations)`;
      report += `${c.name}: ${status}\n`;
    });
    
    report += '\n';
    
    // Distribution Chart
    report += '📊 CALL DISTRIBUTION\n';
    report += '─'.repeat(40) + '\n';
    analytics.fairness.distributionChart.forEach(line => {
      report += line + '\n';
    });
    report += '\n';
    
    // Detailed Resident Stats
    report += '👥 DETAILED RESIDENT STATISTICS\n';
    report += '─'.repeat(40) + '\n';
    report += 'Name'.padEnd(20) + 'PGY  Total  WE  Hol  Night  Day  Backup  Points  Ratio\n';
    report += '─'.repeat(70) + '\n';
    
    analytics.distribution
      .sort((a, b) => b.totalCalls - a.totalCalls)
      .forEach(resident => {
        const name = resident.name.substring(0, 18).padEnd(20);
        const pgy = String(resident.pgyLevel).padEnd(5);
        const total = String(resident.totalCalls).padEnd(7);
        const we = String(resident.weekendCalls).padEnd(4);
        const hol = String(resident.holidayCalls).padEnd(5);
        const night = String(resident.nightCalls).padEnd(7);
        const day = String(resident.dayCalls).padEnd(5);
        const backup = String(resident.backupCalls).padEnd(8);
        const points = String(resident.points.toFixed(1)).padEnd(8);
        const ratio = resident.callRatio;
        
        report += `${name}${pgy}${total}${we}${hol}${night}${day}${backup}${points}${ratio}\n`;
      });
    
    // Violations Detail
    if (validation.violations.length > 0) {
      report += '\n⚠️ VIOLATIONS DETAIL\n';
      report += '─'.repeat(40) + '\n';
      
      validation.violations.forEach((v, i) => {
        report += `${i + 1}. [${v.type.toUpperCase()}] ${v.rule}\n`;
        report += `   ${v.description}\n`;
        report += `   Date: ${v.date.toDateString()}\n\n`;
      });
    }
    
    // Warnings
    if (validation.warnings.length > 0) {
      report += '\n⚠️ WARNINGS\n';
      report += '─'.repeat(40) + '\n';
      
      validation.warnings.forEach((w, i) => {
        report += `${i + 1}. ${w}\n`;
      });
    }
    
    report += '\n' + '═'.repeat(60) + '\n';
    report += `Generated: ${new Date().toISOString()}\n`;
    report += `Scheduler Version: 2.0.0\n`;
    report += '═'.repeat(60) + '\n';
    
    return report;
  }
}

// Export the scheduler
export default MonthlyCallScheduler;]): number {
    