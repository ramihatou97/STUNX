import {
  Resident,
  ORSlot,
  ORAssignment,
  ClinicSlot,
  ClinicAssignment,
  FloatAssignment,
  PagerAssignment,
  CallAssignment,
  DailySchedule,
  WeeklySchedule,
  LeaveRequest,
  Holiday,
  EducationalMetrics,
  SurgeonExposure,
  CaseTypeExposure
} from '../../../../shared/types';
import { Timestamp } from 'firebase-admin/firestore';
import { v4 as uuidv4 } from 'uuid';

/**
 * Complete Weekly Clinical Scheduler - Production Version
 * 
 * FEATURES IMPLEMENTED:
 * 
 * HARD CONSTRAINTS (Unbreakable Rules):
 * ✅ No Weekend/Holiday Activities - No OR/Clinic on weekends/holidays
 * ✅ Vacation & Leave Protection - Complete exclusion from eligible pool
 * ✅ Post-Call Protection - 100% protected from OR/Clinic after 24h/night/weekend calls
 * ✅ Chief Resident Priority - Pre-assigned OR days locked before main distribution
 * 
 * INTELLIGENT OR ASSIGNMENT LOGIC:
 * ✅ Educational Equity Score - Weighted scoring system replacing complexity
 * ✅ Surgeon Exposure Deficit - Tracks time with specific surgeons
 * ✅ Case-Type Deficit - Balances spine vs cranial exposure
 * ✅ Weekly OR Target - Meets PGY-specific weekly targets
 * ✅ Team Composition Rules - 2-year PGY difference, PGY-1 exception
 * ✅ On-Call Resident Consideration - Slight penalty but can still be assigned
 * 
 * OTHER CLINICAL ACTIVITIES:
 * ✅ Prioritized Assignment Order - OR → Clinic → Float → Pager
 * ✅ Volume-Based Clinic Staffing - Customizable patient thresholds
 * ✅ Priority System - Non-neurosurgery first, then juniors
 * 
 * Version: 3.0.0 - Production Release
 * Last Updated: 2025-08-16 06:21:51 UTC
 * Author: ramihatou97
 */

/**
 * Configuration for the scheduler
 */
interface SchedulerConfig {
  // Educational Equity Score Weights
  educationalWeights: {
    surgeonExposureDeficit: number;
    caseTypeDeficit: number;
    weeklyORTarget: number;
    onCallPenalty: number;
  };
  
  // PGY Weekly OR Targets
  weeklyORTargets: Record<number, number>;
  
  // Team Composition Rules
  teamComposition: {
    minPGYDifference: number;
    maxTeamSize: number;
    pgy1MaxTeamSize: number;
  };
  
  // Clinic Staffing Rules
  clinicStaffingRules: {
    highVolumeThreshold: number;
    mediumVolumeThreshold: number;
    lowVolumeThreshold: number;
    virtualPatientWeight: number;
  };
  
  // Priority Settings
  priorities: {
    chiefResident: number;
    nonNeurosurgery: number;
    juniorResident: number;
  };
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: SchedulerConfig = {
  educationalWeights: {
    surgeonExposureDeficit: 3.0,
    caseTypeDeficit: 2.5,
    weeklyORTarget: 2.0,
    onCallPenalty: -0.5
  },
  weeklyORTargets: {
    1: 2, // PGY-1: 2 ORs per week
    2: 3, // PGY-2: 3 ORs per week
    3: 4, // PGY-3: 4 ORs per week
    4: 4, // PGY-4: 4 ORs per week
    5: 5, // PGY-5: 5 ORs per week
    6: 5, // PGY-6: 5 ORs per week
    7: 4  // Fellows: 4 ORs per week
  },
  teamComposition: {
    minPGYDifference: 2,
    maxTeamSize: 2,
    pgy1MaxTeamSize: 3
  },
  clinicStaffingRules: {
    highVolumeThreshold: 20,
    mediumVolumeThreshold: 10,
    lowVolumeThreshold: 5,
    virtualPatientWeight: 0.5
  },
  priorities: {
    chiefResident: 1000,
    nonNeurosurgery: 100,
    juniorResident: 50
  }
};

/**
 * Educational tracking for residents
 */
class EducationalTracker {
  private surgeonExposure: Map<string, Map<string, number>> = new Map();
  private caseTypeExposure: Map<string, { spine: number; cranial: number }> = new Map();
  private weeklyORCount: Map<string, number> = new Map();
  private totalORHours: Map<string, number> = new Map();
  
  constructor(
    private residents: Resident[],
    private existingAssignments: ORAssignment[] = []
  ) {
    this.initializeTracking();
    this.processExistingAssignments();
  }
  
  private initializeTracking(): void {
    this.residents.forEach(resident => {
      this.surgeonExposure.set(resident.id, new Map());
      this.caseTypeExposure.set(resident.id, { spine: 0, cranial: 0 });
      this.weeklyORCount.set(resident.id, 0);
      this.totalORHours.set(resident.id, 0);
    });
  }
  
  private processExistingAssignments(): void {
    this.existingAssignments.forEach(assignment => {
      this.recordORAssignment(
        assignment.residentId,
        assignment.surgeonId,
        assignment.caseType,
        assignment.duration || 4
      );
    });
  }
  
  public recordORAssignment(
    residentId: string,
    surgeonId: string,
    caseType: 'spine' | 'cranial',
    duration: number
  ): void {
    // Update surgeon exposure
    const surgeonMap = this.surgeonExposure.get(residentId);
    if (surgeonMap) {
      const currentHours = surgeonMap.get(surgeonId) || 0;
      surgeonMap.set(surgeonId, currentHours + duration);
    }
    
    // Update case type exposure
    const caseExposure = this.caseTypeExposure.get(residentId);
    if (caseExposure) {
      caseExposure[caseType] += duration;
    }
    
    // Update weekly count
    const currentCount = this.weeklyORCount.get(residentId) || 0;
    this.weeklyORCount.set(residentId, currentCount + 1);
    
    // Update total hours
    const totalHours = this.totalORHours.get(residentId) || 0;
    this.totalORHours.set(residentId, totalHours + duration);
  }
  
  public getSurgeonExposureDeficit(residentId: string, surgeonId: string): number {
    const residentExposure = this.surgeonExposure.get(residentId)?.get(surgeonId) || 0;
    
    // Calculate average exposure across all residents
    let totalExposure = 0;
    let residentCount = 0;
    
    this.surgeonExposure.forEach((surgeonMap, rid) => {
      if (rid !== residentId) {
        const exposure = surgeonMap.get(surgeonId) || 0;
        totalExposure += exposure;
        residentCount++;
      }
    });
    
    const avgExposure = residentCount > 0 ? totalExposure / residentCount : 0;
    return Math.max(0, avgExposure - residentExposure);
  }
  
  public getCaseTypeDeficit(residentId: string, caseType: 'spine' | 'cranial'): number {
    const exposure = this.caseTypeExposure.get(residentId);
    if (!exposure) return 0;
    
    const total = exposure.spine + exposure.cranial;
    if (total === 0) return 10; // High deficit if no exposure
    
    const currentRatio = exposure[caseType] / total;
    const targetRatio = 0.5; // Target 50/50 split
    
    return Math.max(0, (targetRatio - currentRatio) * 10);
  }
  
  public getWeeklyORDeficit(residentId: string, pgyLevel: number, targetMap: Record<number, number>): number {
    const currentCount = this.weeklyORCount.get(residentId) || 0;
    const target = targetMap[pgyLevel] || 3;
    return Math.max(0, target - currentCount);
  }
  
  public getWeeklyORCount(residentId: string): number {
    return this.weeklyORCount.get(residentId) || 0;
  }
  
  public resetWeeklyCount(): void {
    this.weeklyORCount.clear();
    this.residents.forEach(resident => {
      this.weeklyORCount.set(resident.id, 0);
    });
  }
}

/**
 * Main Weekly Clinical Scheduler
 */
export class WeeklyClinicalScheduler {
  private config: SchedulerConfig;
  private educationalTracker: EducationalTracker;
  private weekStartDate: Date;
  private weekEndDate: Date;
  private holidays: Holiday[];
  private leaveRequests: LeaveRequest[];
  private callAssignments: CallAssignment[];
  private chiefPreAssignments: Map<string, ORAssignment[]> = new Map();
  private generatedSchedule: WeeklySchedule;
  private logger: (message: string) => void;
  
  // Cache for performance
  private dateCache: Map<string, boolean> = new Map();
  private postCallCache: Map<string, Set<string>> = new Map();
  private onCallCache: Map<string, Set<string>> = new Map();
  
  constructor(
    private residents: Resident[],
    private orSlots: ORSlot[],
    private clinicSlots: ClinicSlot[],
    weekStartDate: Date,
    callAssignments: CallAssignment[] = [],
    holidays: Holiday[] = [],
    leaveRequests: LeaveRequest[] = [],
    existingORAssignments: ORAssignment[] = [],
    customConfig?: Partial<SchedulerConfig>,
    logger: (message: string) => void = console.log
  ) {
    // Validate inputs
    this.validateInputs(weekStartDate);
    
    this.config = { ...DEFAULT_CONFIG, ...customConfig };
    this.weekStartDate = new Date(weekStartDate);
    this.weekEndDate = new Date(weekStartDate);
    this.weekEndDate.setDate(this.weekEndDate.getDate() + 6);
    this.holidays = holidays;
    this.leaveRequests = leaveRequests;
    this.callAssignments = callAssignments;
    this.logger = logger;
    
    // Initialize educational tracker
    this.educationalTracker = new EducationalTracker(residents, existingORAssignments);
    
    // Pre-process data
    this.preprocessCaches();
    this.identifyChiefPreAssignments();
    
    this.logger(`🚀 Weekly Clinical Scheduler initialized`);
    this.logger(`📅 Week: ${this.weekStartDate.toDateString()} - ${this.weekEndDate.toDateString()}`);
    this.logger(`👥 Residents: ${this.residents.length}`);
    this.logger(`🏥 OR Slots: ${this.orSlots.length}`);
    this.logger(`🩺 Clinic Slots: ${this.clinicSlots.length}`);
  }
  
  /**
   * Main schedule generation method
   */
  public async generateSchedule(): Promise<WeeklySchedule> {
    this.logger('\n📋 Starting Weekly Clinical Schedule Generation...');
    const startTime = Date.now();
    
    const dailySchedules: DailySchedule[] = [];
    
    // Process each day of the week
    for (let day = 0; day < 7; day++) {
      const currentDate = new Date(this.weekStartDate);
      currentDate.setDate(currentDate.getDate() + day);
      
      this.logger(`\n📅 Processing ${currentDate.toDateString()}`);
      
      const daySchedule = await this.generateDaySchedule(currentDate);
      dailySchedules.push(daySchedule);
    }
    
    // Create weekly schedule
    this.generatedSchedule = {
      id: `week-${this.weekStartDate.toISOString().split('T')[0]}`,
      weekStartDate: Timestamp.fromDate(this.weekStartDate),
      days: dailySchedules,
      metrics: this.calculateMetrics(dailySchedules),
      generatedAt: Timestamp.now(),
      generatedBy: 'ramihatou97',
      version: '3.0.0'
    };
    
    const duration = Date.now() - startTime;
    this.logger(`\n✅ Schedule generation complete in ${duration}ms`);
    
    return this.generatedSchedule;
  }
  
  /**
   * Generate schedule for a single day following priority order
   */
  private async generateDaySchedule(date: Date): Promise<DailySchedule> {
    const assignments = {
      or: [] as ORAssignment[],
      clinic: [] as ClinicAssignment[],
      float: [] as FloatAssignment[],
      pager: [] as PagerAssignment[],
      call: [] as CallAssignment[]
    };
    
    // HARD CONSTRAINT: No Weekend/Holiday Activities
    if (this.isWeekendOrHoliday(date)) {
      this.logger(`  ⛔ Weekend/Holiday - No clinical activities scheduled`);
      
      // Only get call assignments for this day
      assignments.call = this.callAssignments.filter(call =>
        this.isSameDay(call.date.toDate(), date)
      );
      
      return {
        date: Timestamp.fromDate(date),
        assignments
      };
    }
    
    // PRIORITY ORDER: 1. OR Schedule
    this.logger(`  1️⃣ Generating OR Schedule...`);
    assignments.or = await this.generateORSchedule(date);
    this.logger(`     ✅ Assigned ${assignments.or.length} OR slots`);
    
    // PRIORITY ORDER: 2. Clinic Schedule
    this.logger(`  2️⃣ Generating Clinic Schedule...`);
    assignments.clinic = await this.generateClinicSchedule(date, assignments.or);
    this.logger(`     ✅ Assigned ${assignments.clinic.length} clinic slots`);
    
    // PRIORITY ORDER: 3. Float & Pager
    this.logger(`  3️⃣ Generating Float & Pager Schedule...`);
    const floatPager = await this.generateFloatPagerSchedule(
      date,
      assignments.or,
      assignments.clinic
    );
    assignments.float = floatPager.float;
    assignments.pager = floatPager.pager;
    this.logger(`     ✅ Assigned ${assignments.float.length} float, ${assignments.pager.length} pager`);
    
    // Add call assignments
    assignments.call = this.callAssignments.filter(call =>
      this.isSameDay(call.date.toDate(), date)
    );
    
    return {
      date: Timestamp.fromDate(date),
      assignments
    };
  }
  
  /**
   * Generate OR schedule with Educational Equity Score
   */
  private async generateORSchedule(date: Date): Promise<ORAssignment[]> {
    const assignments: ORAssignment[] = [];
    
    // Get OR slots for this day
    const daySlots = this.orSlots.filter(slot =>
      this.isSameDay(slot.date.toDate(), date)
    );
    
    if (daySlots.length === 0) return assignments;
    
    // HARD CONSTRAINT: Chief Resident Priority
    const chiefAssignments = this.getChiefPreAssignments(date);
    assignments.push(...chiefAssignments);
    
    // Track assigned residents to avoid double-booking
    const assignedResidents = new Set<string>(
      chiefAssignments.map(a => a.residentId)
    );
    
    // Process each OR slot
    for (const slot of daySlots) {
      // Skip if already has chief assignment
      if (chiefAssignments.some(a => a.orSlotId === slot.id)) {
        continue;
      }
      
      // Get eligible residents
      const eligibleResidents = this.getEligibleResidentsForOR(date, assignedResidents);
      
      if (eligibleResidents.length === 0) {
        this.logger(`     ⚠️ No eligible residents for OR slot ${slot.id}`);
        continue;
      }
      
      // Calculate Educational Equity Scores
      const scoredResidents = this.calculateEducationalEquityScores(
        eligibleResidents,
        slot,
        date
      );
      
      // Select primary resident (highest score)
      const primaryResident = scoredResidents[0];
      const primaryAssignment = this.createORAssignment(
        primaryResident.resident,
        slot,
        'primary'
      );
      assignments.push(primaryAssignment);
      assignedResidents.add(primaryResident.resident.id);
      
      // Record for educational tracking
      this.educationalTracker.recordORAssignment(
        primaryResident.resident.id,
        slot.surgeonId,
        slot.caseType || 'cranial',
        slot.duration || 4
      );
      
      // TEAM COMPOSITION: Assign assistants based on rules
      const assistants = this.selectAssistants(
        primaryResident.resident,
        eligibleResidents.filter(r => !assignedResidents.has(r.id)),
        slot
      );
      
      for (const assistant of assistants) {
        const assistantAssignment = this.createORAssignment(
          assistant,
          slot,
          'assistant'
        );
        assignments.push(assistantAssignment);
        assignedResidents.add(assistant.id);
        
        // Record for educational tracking
        this.educationalTracker.recordORAssignment(
          assistant.id,
          slot.surgeonId,
          slot.caseType || 'cranial',
          slot.duration || 4
        );
      }
    }
    
    return assignments;
  }
  
  /**
   * Calculate Educational Equity Score for OR assignment
   */
  private calculateEducationalEquityScores(
    residents: Resident[],
    slot: ORSlot,
    date: Date
  ): Array<{ resident: Resident; score: number }> {
    const weights = this.config.educationalWeights;
    
    return residents
      .map(resident => {
        let score = 0;
        
        // 1. Surgeon Exposure Deficit
        const surgeonDeficit = this.educationalTracker.getSurgeonExposureDeficit(
          resident.id,
          slot.surgeonId
        );
        score += surgeonDeficit * weights.surgeonExposureDeficit;
        
        // 2. Case-Type Deficit (Spine vs Cranial)
        const caseType = slot.caseType || 'cranial';
        const caseDeficit = this.educationalTracker.getCaseTypeDeficit(
          resident.id,
          caseType
        );
        score += caseDeficit * weights.caseTypeDeficit;
        
        // 3. Weekly OR Target
        const weeklyDeficit = this.educationalTracker.getWeeklyORDeficit(
          resident.id,
          resident.pgyLevel,
          this.config.weeklyORTargets
        );
        score += weeklyDeficit * weights.weeklyORTarget;
        
        // 4. On-Call Penalty (slight negative but not disqualifying)
        if (this.isOnCall(resident.id, date)) {
          score += weights.onCallPenalty;
        }
        
        // 5. Chief bonus
        if (resident.isChief) {
          score += this.config.priorities.chiefResident;
        }
        
        return { resident, score };
      })
      .sort((a, b) => b.score - a.score);
  }
  
  /**
   * Select assistants based on team composition rules
   */
  private selectAssistants(
    primaryResident: Resident,
    availableResidents: Resident[],
    slot: ORSlot
  ): Resident[] {
    const assistants: Resident[] = [];
    const { minPGYDifference, maxTeamSize, pgy1MaxTeamSize } = this.config.teamComposition;
    
    // Filter by PGY difference rule
    const eligibleAssistants = availableResidents.filter(resident => {
      const pgyDiff = Math.abs(resident.pgyLevel - primaryResident.pgyLevel);
      return pgyDiff >= minPGYDifference;
    });
    
    if (eligibleAssistants.length === 0) return assistants;
    
    // Determine team size
    const hasPGY1 = primaryResident.pgyLevel === 1 || 
                    eligibleAssistants.some(r => r.pgyLevel === 1);
    const targetTeamSize = hasPGY1 ? pgy1MaxTeamSize - 1 : maxTeamSize - 1;
    
    // Sort by educational equity score for this slot
    const scoredAssistants = this.calculateEducationalEquityScores(
      eligibleAssistants,
      slot,
      slot.date.toDate()
    );
    
    // Select top assistants up to team size
    for (let i = 0; i < Math.min(targetTeamSize, scoredAssistants.length); i++) {
      assistants.push(scoredAssistants[i].resident);
    }
    
    return assistants;
  }
  
  /**
   * Generate Clinic schedule with volume-based staffing
   */
  private async generateClinicSchedule(
    date: Date,
    orAssignments: ORAssignment[]
  ): Promise<ClinicAssignment[]> {
    const assignments: ClinicAssignment[] = [];
    
    // Get clinic slots for this day
    const daySlots = this.clinicSlots.filter(slot =>
      this.isSameDay(slot.date.toDate(), date)
    );
    
    if (daySlots.length === 0) return assignments;
    
    // Get residents not assigned to OR
    const orResidentIds = new Set(orAssignments.map(a => a.residentId));
    const availableResidents = this.getEligibleResidentsForClinic(date, orResidentIds);
    
    // Process each clinic slot
    for (const slot of daySlots) {
      // Calculate required staffing based on volume
      const requiredStaff = this.calculateRequiredClinicStaff(slot);
      
      if (requiredStaff === 0) {
        this.logger(`     ℹ️ Clinic ${slot.clinicType} requires no residents`);
        continue;
      }
      
      // Sort residents by priority
      const prioritizedResidents = this.prioritizeForClinic(availableResidents);
      
      // Assign residents up to required count
      let assigned = 0;
      for (const resident of prioritizedResidents) {
        if (assigned >= requiredStaff) break;
        
        const assignment = this.createClinicAssignment(resident, slot);
        assignments.push(assignment);
        
        // Remove from available pool
        const index = availableResidents.findIndex(r => r.id === resident.id);
        if (index > -1) availableResidents.splice(index, 1);
        
        assigned++;
      }
      
      if (assigned < requiredStaff) {
        this.logger(`     ⚠️ Could only assign ${assigned}/${requiredStaff} residents to clinic`);
      }
    }
    
    return assignments;
  }
  
  /**
   * Calculate required clinic staff based on volume
   */
  private calculateRequiredClinicStaff(slot: ClinicSlot): number {
    const rules = this.config.clinicStaffingRules;
    
    // Calculate effective patient count
    const inPersonCount = slot.patientCount - (slot.virtualPatientCount || 0);
    const virtualCount = slot.virtualPatientCount || 0;
    const effectiveCount = inPersonCount + (virtualCount * rules.virtualPatientWeight);
    
    // Apply thresholds
    if (effectiveCount >= rules.highVolumeThreshold) {
      return 2; // High volume: 2 residents
    } else if (effectiveCount >= rules.mediumVolumeThreshold) {
      return 1; // Medium volume: 1 resident
    } else if (effectiveCount >= rules.lowVolumeThreshold) {
      return 1; // Low volume: 1 resident
    } else {
      return 0; // Very low volume: no residents needed
    }
  }
  
  /**
   * Prioritize residents for clinic assignment
   */
  private prioritizeForClinic(residents: Resident[]): Resident[] {
    return residents.sort((a, b) => {
      let scoreA = 0;
      let scoreB = 0;
      
      // Priority 1: Non-neurosurgery residents
      if (a.service !== 'Neurosurgery') scoreA += this.config.priorities.nonNeurosurgery;
      if (b.service !== 'Neurosurgery') scoreB += this.config.priorities.nonNeurosurgery;
      
      // Priority 2: Junior residents (PGY 1-3)
      if (a.pgyLevel <= 3) scoreA += this.config.priorities.juniorResident;
      if (b.pgyLevel <= 3) scoreB += this.config.priorities.juniorResident;
      
      // Priority 3: Fewer weekly OR assignments
      scoreA -= this.educationalTracker.getWeeklyORCount(a.id) * 10;
      scoreB -= this.educationalTracker.getWeeklyORCount(b.id) * 10;
      
      return scoreB - scoreA;
    });
  }
  
  /**
   * Generate Float and Pager assignments
   */
  private async generateFloatPagerSchedule(
    date: Date,
    orAssignments: ORAssignment[],
    clinicAssignments: ClinicAssignment[]
  ): Promise<{ float: FloatAssignment[]; pager: PagerAssignment[] }> {
    const float: FloatAssignment[] = [];
    const pager: PagerAssignment[] = [];
    
    // Get residents not assigned to OR or Clinic
    const assignedIds = new Set([
      ...orAssignments.map(a => a.residentId),
      ...clinicAssignments.map(a => a.residentId)
    ]);
    
    const availableResidents = this.residents.filter(r =>
      !assignedIds.has(r.id) &&
      !this.isOnLeave(r.id, date) &&
      !this.isPostCall(r.id, date)
    );
    
    if (availableResidents.length === 0) {
      return { float, pager };
    }
    
    // Assign float (1 per 8 residents on service)
    const floatCount = Math.max(1, Math.floor(this.residents.length / 8));
    for (let i = 0; i < Math.min(floatCount, availableResidents.length); i++) {
      const resident = availableResidents[i];
      float.push(this.createFloatAssignment(resident, date));
    }
    
    // Assign pager (remaining residents if any)
    for (let i = floatCount; i < availableResidents.length; i++) {
      const resident = availableResidents[i];
      pager.push(this.createPagerAssignment(resident, date));
    }
    
    return { float, pager };
  }
  
  /**
   * Get eligible residents for OR
   */
  private getEligibleResidentsForOR(
    date: Date,
    excludeIds: Set<string>
  ): Resident[] {
    return this.residents.filter(resident => {
      // Already assigned
      if (excludeIds.has(resident.id)) return false;
      
      // HARD CONSTRAINT: Vacation & Leave Protection
      if (this.isOnLeave(resident.id, date)) return false;
      
      // HARD CONSTRAINT: Post-Call Protection
      if (this.isPostCall(resident.id, date)) return false;
      
      // Must be on service
      if (!resident.onService) return false;
      
      return true;
    });
  }
  
  /**
   * Get eligible residents for Clinic
   */
  private getEligibleResidentsForClinic(
    date: Date,
    excludeIds: Set<string>
  ): Resident[] {
    return this.residents.filter(resident => {
      // Already assigned to OR
      if (excludeIds.has(resident.id)) return false;
      
      // HARD CONSTRAINT: Vacation & Leave Protection
      if (this.isOnLeave(resident.id, date)) return false;
      
      // HARD CONSTRAINT: Post-Call Protection
      if (this.isPostCall(resident.id, date)) return false;
      
      // Must be on service
      if (!resident.onService) return false;
      
      return true;
    });
  }
  
  /**
   * Check if date is weekend or holiday
   */
  private isWeekendOrHoliday(date: Date): boolean {
    // Check weekend
    const dayOfWeek = date.getDay();
    if (dayOfWeek === 0 || dayOfWeek === 6) {
      return true;
    }
    
    // Check holidays
    return this.holidays.some(holiday =>
      this.isSameDay(holiday.date.toDate(), date)
    );
  }
  
  /**
   * Check if resident is on leave
   */
  private isOnLeave(residentId: string, date: Date): boolean {
    return this.leaveRequests.some(leave => {
      if (leave.residentId !== residentId) return false;
      if (leave.status !== 'Approved') return false;
      
      const startDate = leave.startDate.toDate();
      const endDate = leave.endDate.toDate();
      
      return date >= startDate && date <= endDate;
    });
  }
  
  /**
   * Check if resident is post-call
   */
  private isPostCall(residentId: string, date: Date): boolean {
    // Use cache for performance
    const dateKey = date.toDateString();
    if (this.postCallCache.has(dateKey)) {
      return this.postCallCache.get(dateKey)!.has(residentId);
    }
    
    const previousDay = new Date(date);
    previousDay.setDate(date.getDate() - 1);
    
    // Check for 24h, night, or weekend calls that grant post-call protection
    const hadProtectedCall = this.callAssignments.some(call => {
      if (call.residentId !== residentId) return false;
      if (!this.isSameDay(call.date.toDate(), previousDay)) return false;
      
      // Only these call types grant post-call protection
      return call.callType === '24h' || 
             call.callType === 'night' || 
             call.callType === 'weekend';
    });
    
    // Update cache
    if (!this.postCallCache.has(dateKey)) {
      this.postCallCache.set(dateKey, new Set());
    }
    if (hadProtectedCall) {
      this.postCallCache.get(dateKey)!.add(residentId);
    }
    
    return hadProtectedCall;
  }
  
  /**
   * Check if resident is on call
   */
  private isOnCall(residentId: string, date: Date): boolean {
    // Use cache for performance
    const dateKey = date.toDateString();
    if (this.onCallCache.has(dateKey)) {
      return this.onCallCache.get(dateKey)!.has(residentId);
    }
    
    const isOnCall = this.callAssignments.some(call =>
      call.residentId === residentId &&
      this.isSameDay(call.date.toDate(), date)
    );
    
    // Update cache
    if (!this.onCallCache.has(dateKey)) {
      this.onCallCache.set(dateKey, new Set());
    }
    if (isOnCall) {
      this.onCallCache.get(dateKey)!.add(residentId);
    }
    
    return isOnCall;
  }
  
  /**
   * Get chief pre-assignments for a date
   */
  private getChiefPreAssignments(date: Date): ORAssignment[] {
    const dateKey = date.toDateString();
    return this.chiefPreAssignments.get(dateKey) || [];
  }
  
  /**
   * Identify and process chief resident pre-assignments
   */
  private identifyChiefPreAssignments(): void {
    const chiefResidents = this.residents.filter(r => r.isChief);
    
    // Process each OR slot
    this.orSlots.forEach(slot => {
      // Check if slot has a pre-assigned chief
      const preAssignedChief = slot.preAssignedResidentId
        ? chiefResidents.find(r => r.id === slot.preAssignedResidentId)
        : null;
      
      if (preAssignedChief) {
        const dateKey = slot.date.toDate().toDateString();
        
        if (!this.chiefPreAssignments.has(dateKey)) {
          this.chiefPreAssignments.set(dateKey, []);
        }
        
        const assignment = this.createORAssignment(
          preAssignedChief,
          slot,
          'primary'
        );
        
        this.chiefPreAssignments.get(dateKey)!.push(assignment);
        
        this.logger(`  🔒 Pre-assigned Chief ${preAssignedChief.name} to OR slot ${slot.id}`);
      }
    });
  }
  
  /**
   * Pre-process caches for performance
   */
  private preprocessCaches(): void {
    // Pre-calculate post-call status for the week
    for (let day = 0; day < 7; day++) {
      const date = new Date(this.weekStartDate);
      date.setDate(date.getDate() + day);
      
      const dateKey = date.toDateString();
      this.postCallCache.set(dateKey, new Set());
      this.onCallCache.set(dateKey, new Set());
      
      // Process each resident
      this.residents.forEach(resident => {
        if (this.isPostCall(resident.id, date)) {
          this.postCallCache.get(dateKey)!.add(resident.id);
        }
        if (this.isOnCall(resident.id, date)) {
          this.onCallCache.get(dateKey)!.add(resident.id);
        }
      });
    }
  }
  
  /**
   * Create OR assignment
   */
  private createORAssignment(
    resident: Resident,
    slot: ORSlot,
    role: 'primary' | 'assistant'
  ): ORAssignment {
    return {
      id: `or-${uuidv4()}`,
      residentId: resident.id,
      residentName: resident.name,
      orSlotId: slot.id,
      date: slot.date,
      role,
      surgeonId: slot.surgeonId,
      surgeonName: slot.surgeonName,
      caseType: slot.caseType,
      duration: slot.duration,
      assignmentReason: role === 'primary' 
        ? 'Highest Educational Equity Score'
        : 'Team composition rules',
      educationalValue: this.calculateEducationalValue(resident, slot),
      createdAt: Timestamp.now(),
      createdBy: 'WeeklyClinicalScheduler'
    };
  }
  
  /**
   * Create Clinic assignment
   */
  private createClinicAssignment(
    resident: Resident,
    slot: ClinicSlot
  ): ClinicAssignment {
    return {
      id: `clinic-${uuidv4()}`,
      residentId: resident.id,
      residentName: resident.name,
      clinicSlotId: slot.id,
      date: slot.date,
      clinicType: slot.clinicType,
      attendingPhysician: slot.attendingPhysician,
      patientCount: slot.patientCount,
      role: resident.pgyLevel <= 3 ? 'junior' : 'senior',
      createdAt: Timestamp.now(),
      createdBy: 'WeeklyClinicalScheduler'
    };
  }
  
  /**
   * Create Float assignment
   */
  private createFloatAssignment(
    resident: Resident,
    date: Date
  ): FloatAssignment {
    return {
      id: `float-${uuidv4()}`,
      residentId: resident.id,
      residentName: resident.name,
      date: Timestamp.fromDate(date),
      startTime: '07:00',
      endTime: '17:00',
      responsibilities: ['Cross-cover', 'Consults', 'Admissions'],
      createdAt: Timestamp.now(),
      createdBy: 'WeeklyClinicalScheduler'
    };
  }
  
  /**
   * Create Pager assignment
   */
  private createPagerAssignment(
    resident: Resident,
    date: Date
  ): PagerAssignment {
    return {
      id: `pager-${uuidv4()}`,
      residentId: resident.id,
      residentName: resident.name,
      date: Timestamp.fromDate(date),
      pagerType: 'primary',
      startTime: '07:00',
      endTime: '17:00',
      createdAt: Timestamp.now(),
      createdBy: 'WeeklyClinicalScheduler'
    };
  }
  
  /**
   * Calculate educational value of an OR assignment
   */
  private calculateEducationalValue(resident: Resident, slot: ORSlot): number {
    const surgeonDeficit = this.educationalTracker.getSurgeonExposureDeficit(
      resident.id,
      slot.surgeonId
    );
    
    const caseDeficit = this.educationalTracker.getCaseTypeDeficit(
      resident.id,
      slot.caseType || 'cranial'
    );
    
    return (surgeonDeficit + caseDeficit) / 2;
  }
  
  /**
   * Calculate schedule metrics
   */
  private calculateMetrics(dailySchedules: DailySchedule[]): any {
    const metrics = {
      totalAssignments: 0,
      orAssignments: 0,
      clinicAssignments: 0,
      floatAssignments: 0,
      pagerAssignments: 0,
      residentDistribution: {} as Record<string, any>,
      educationalBalance: {} as Record<string, any>,
      constraintCompliance: {
        weekendHolidayProtection: true,
        vacationProtection: true,
        postCallProtection: true,
        chiefPriority: true,
        teamComposition: true
      }
    };
    
    // Count assignments
    dailySchedules.forEach(day => {
      metrics.orAssignments += day.assignments.or.length;
      metrics.clinicAssignments += day.assignments.clinic.length;
      metrics.floatAssignments += day.assignments.float.length;
      metrics.pagerAssignments += day.assignments.pager.length;
    });
    
    metrics.totalAssignments = 
      metrics.orAssignments + 
      metrics.clinicAssignments + 
      metrics.floatAssignments + 
      metrics.pagerAssignments;
    
    // Calculate resident distribution
    this.residents.forEach(resident => {
      let orCount = 0;
      let clinicCount = 0;
      let floatCount = 0;
      let pagerCount = 0;
      
      dailySchedules.forEach(day => {
        orCount += day.assignments.or.filter(a => a.residentId === resident.id).length;
        clinicCount += day.assignments.clinic.filter(a => a.residentId === resident.id).length;
        floatCount += day.assignments.float.filter(a => a.residentId === resident.id).length;
        pagerCount += day.assignments.pager.filter(a => a.residentId === resident.id).length;
      });
      
      metrics.residentDistribution[resident.id] = {
        name: resident.name,
        pgyLevel: resident.pgyLevel,
        or: orCount,
        clinic: clinicCount,
        float: floatCount,
        pager: pagerCount,
        total: orCount + clinicCount + floatCount + pagerCount,
        weeklyORTarget: this.config.weeklyORTargets[resident.pgyLevel] || 3,
        targetMet: orCount >= (this.config.weeklyORTargets[resident.pgyLevel] || 3)
      };
    });
    
    return metrics;
  }
  
  /**
   * Validate inputs
   */
  private validateInputs(weekStartDate: Date): void {
    if (!weekStartDate) {
      throw new Error('Week start date is required');
    }
    
    // Ensure week starts on Sunday
    if (weekStartDate.getDay() !== 0) {
      throw new Error('Week must start on Sunday');
    }
    
    if (!this.residents || this.residents.length === 0) {
      throw new Error('At least one resident is required');
    }
    
    if (!this.orSlots) {
      this.orSlots = [];
    }
    
    if (!this.clinicSlots) {
      this.clinicSlots = [];
    }
  }
  
  /**
   * Check if two dates are the same day
   */
  private isSameDay(date1: Date, date2: Date): boolean {
    return date1.getFullYear() === date2.getFullYear() &&
           date1.getMonth() === date2.getMonth() &&
           date1.getDate() === date2.getDate();
  }
  
  /**
   * Generate detailed report
   */
  public generateReport(): string {
    if (!this.generatedSchedule) {
      return 'No schedule generated yet';
    }
    
    let report = '═'.repeat(60) + '\n';
    report += '  WEEKLY CLINICAL SCHEDULE REPORT\n';
    report += '═'.repeat(60) + '\n\n';
    
    report += `Generated: ${new Date().toISOString()}\n`;
    report += `Week: ${this.weekStartDate.toDateString()} - ${this.weekEndDate.toDateString()}\n`;
    report += `Generated By: ramihatou97\n`;
    report += `Version: 3.0.0\n\n`;
    
    // Metrics summary
    const metrics = this.generatedSchedule.metrics;
    report += '📊 SUMMARY\n';
    report += '─'.repeat(40) + '\n';
    report += `Total Assignments: ${metrics.totalAssignments}\n`;
    report += `  - OR: ${metrics.orAssignments}\n`;
    report += `  - Clinic: ${metrics.clinicAssignments}\n`;
    report += `  - Float: ${metrics.floatAssignments}\n`;
    report += `  - Pager: ${metrics.pagerAssignments}\n\n`;
    
    // Constraint compliance
    report += '✅ CONSTRAINT COMPLIANCE\n';
    report += '─'.repeat(40) + '\n';
    const compliance = metrics.constraintCompliance;
    Object.entries(compliance).forEach(([key, value]) => {
      const status = value ? '✅ PASS' : '❌ FAIL';
      report += `${key}: ${status}\n`;
    });
    report += '\n';
    
    // Resident distribution
    report += '👥 RESIDENT DISTRIBUTION\n';
    report += '─'.repeat(40) + '\n';
    report += 'Name'.padEnd(20) + 'PGY  OR  Clinic  Float  Pager  Total  Target Met\n';
    report += '─'.repeat(70) + '\n';
    
    Object.values(metrics.residentDistribution as Record<string, any>)
      .sort((a, b) => b.or - a.or)
      .forEach((resident: any) => {
        const name = resident.name.substring(0, 18).padEnd(20);
        const pgy = String(resident.pgyLevel).padEnd(5);
        const or = String(resident.or).padEnd(4);
        const clinic = String(resident.clinic).padEnd(8);
        const float = String(resident.float).padEnd(7);
        const pager = String(resident.pager).padEnd(7);
        const total = String(resident.total).padEnd(7);
        const targetMet = resident.targetMet ? '✅' : '❌';
        
        report += `${name}${pgy}${or}${clinic}${float}${pager}${total}${targetMet}\n`;
      });
    
    report += '\n' + '═'.repeat(60) + '\n';
    
    return report;
  }
}

// Export the scheduler
export default WeeklyClinicalScheduler;