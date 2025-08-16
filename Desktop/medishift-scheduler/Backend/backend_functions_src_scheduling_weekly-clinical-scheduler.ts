import { 
    Resident, 
    CallAssignment, 
    AppConfiguration, 
    AcademicYear,
    LeaveRequest,
    ORSchedule,
    ClinicSchedule,
    ClinicalAssignment,
    Surgeon,
    CaseType,
    EducationalTarget,
    ClinicalActivity
} from '../../../shared/types';
import * as admin from 'firebase-admin';
import { v4 as uuidv4 } from 'uuid';
import { MonthlyCallScheduler } from './monthly-scheduler_v10_complete';

/**
 * Educational tracking for residents
 */
interface ResidentEducationalProfile {
    residentId: string;
    surgeonExposureHours: Map<string, number>; // surgeonId -> hours
    caseTypeHours: Map<CaseType, number>; // Spine vs Cranial hours
    weeklyORCount: number;
    totalORHours: number;
    clinicHours: number;
    lastORDate: Date | null;
    educationalDeficits: {
        surgeonDeficit: number;
        caseTypeDeficit: number;
        weeklyTargetDeficit: number;
    };
}

/**
 * OR Assignment with team composition
 */
interface ORAssignment extends ClinicalAssignment {
    surgeonId: string;
    surgeonName: string;
    caseType: CaseType;
    isPrimary: boolean;
    teamMembers: string[]; // Other resident IDs in the team
    estimatedDuration: number; // in hours
    educationalScore: number;
    chiefOverride?: boolean;
}

/**
 * Clinic Assignment with volume tracking
 */
interface ClinicAssignment extends ClinicalAssignment {
    clinicType: string;
    expectedPatients: number;
    requiredStaffing: number;
    assignedResidents: string[];
}

/**
 * Educational Equity Score Components
 */
interface EducationalEquityScore {
    total: number;
    surgeonExposureDeficit: number;
    caseTypeDeficit: number;
    weeklyORTargetDeficit: number;
    onCallPenalty: number;
    breakdown: string;
}

/**
 * Weekly schedule generation metrics
 */
interface WeeklyScheduleMetrics {
    orAssignments: number;
    clinicAssignments: number;
    floatAssignments: number;
    pagerAssignments: number;
    educationalEquityAverage: number;
    violationsCount: number;
    generationTimeMs: number;
}

/**
 * Complete Weekly Clinical Scheduler
 * 
 * Implements all scheduling logic rules:
 * - Hard Constraints: Weekend/holiday protection, Leave protection, Post-call protection, Chief priority
 * - Intelligent OR Assignment: Educational Equity Score, Team composition, On-call consideration
 * - Clinical Activities: Prioritized assignment order, Volume-based clinic staffing
 * 
 * Created: 2025-08-16 05:45:00 UTC by ramihatou97
 * Version: 1.0.0 (Production Release)
 */
export class WeeklyClinicalScheduler {
    private residents: Resident[];
    private config: AppConfiguration;
    private academicYear: AcademicYear;
    private approvedLeave: LeaveRequest[];
    private callSchedule: CallAssignment[];
    private weekStartDate: Date;
    private weekEndDate: Date;
    private surgeons: Surgeon[];
    private educationalProfiles: Map<string, ResidentEducationalProfile>;
    private orSchedules: Map<string, ORSchedule[]>; // date string -> OR schedules
    private clinicSchedules: Map<string, ClinicSchedule[]>; // date string -> clinic schedules
    private generatedAssignments: ClinicalAssignment[] = [];
    private metrics: WeeklyScheduleMetrics;
    private violations: string[] = [];
    
    // PGY-based weekly OR targets
    private readonly WEEKLY_OR_TARGETS: Record<number, number> = {
        1: 2, // PGY-1: 2 ORs per week
        2: 3, // PGY-2: 3 ORs per week
        3: 4, // PGY-3: 4 ORs per week
        4: 4, // PGY-4: 4 ORs per week
        5: 5, // PGY-5: 5 ORs per week
        6: 5, // Fellows: 5 ORs per week
        7: 5  // Senior Fellows: 5 ORs per week
    };
    
    // Clinic staffing rules (patient count -> required residents)
    private readonly CLINIC_STAFFING_RULES = [
        { minPatients: 30, requiredStaff: 3 },
        { minPatients: 20, requiredStaff: 2 },
        { minPatients: 10, requiredStaff: 1 },
        { minPatients: 0, requiredStaff: 1 }
    ];
    

    // Educational Equity Score weights (customizable)
    private readonly SCORE_WEIGHTS: {
        surgeonExposure: number;
        caseType: number;
        weeklyTarget: number;
        onCallPenalty: number;
    };

    // Customizable case types for educational tracking
    private readonly CASE_TYPES: CaseType[];

    /**
     * @param residents List of residents
     * @param config App configuration
     * @param academicYear Academic year
     * @param approvedLeave Approved leave requests
     * @param callSchedule Call assignments
     * @param weekStartDate Start date of the week
     * @param surgeons List of surgeons
     * @param existingORSchedules Existing OR schedules
     * @param existingClinicSchedules Existing clinic schedules
     * @param customCaseTypes (Optional) Custom list of case types for tracking
     * @param customScoreWeights (Optional) Custom score weights for educational equity
     */
    constructor(
        residents: Resident[],
        config: AppConfiguration,
        academicYear: AcademicYear,
        approvedLeave: LeaveRequest[],
        callSchedule: CallAssignment[],
        weekStartDate: Date,
        surgeons: Surgeon[],
        existingORSchedules: ORSchedule[] = [],
        existingClinicSchedules: ClinicSchedule[] = [],
        customCaseTypes?: CaseType[],
        customScoreWeights?: { surgeonExposure: number; caseType: number; weeklyTarget: number; onCallPenalty: number; }
    ) {
        this.residents = residents;
        this.config = config;
        this.academicYear = academicYear;
        this.approvedLeave = approvedLeave;
        this.callSchedule = callSchedule;
        this.weekStartDate = new Date(weekStartDate);
        this.weekEndDate = new Date(weekStartDate);
        this.weekEndDate.setDate(this.weekEndDate.getDate() + 6);
        this.surgeons = surgeons;

        // Use custom or default case types
        this.CASE_TYPES = customCaseTypes && customCaseTypes.length > 0
            ? customCaseTypes
            : ['Spine', 'Cranial', 'Pediatric', 'Functional', 'Vascular'];

        // Use custom or default score weights
        this.SCORE_WEIGHTS = customScoreWeights || {
            surgeonExposure: 0.35,
            caseType: 0.35,
            weeklyTarget: 0.20,
            onCallPenalty: -0.10
        };

        console.log(`🏥 Initializing Weekly Clinical Scheduler`);
        console.log(`📅 Week: ${this.weekStartDate.toDateString()} - ${this.weekEndDate.toDateString()}`);
        console.log(`👥 Residents: ${this.residents.length}`);
        console.log(`🔬 Surgeons: ${this.surgeons.length}`);

        this.educationalProfiles = new Map();
        this.orSchedules = new Map();
        this.clinicSchedules = new Map();

        this.metrics = {
            orAssignments: 0,
            clinicAssignments: 0,
            floatAssignments: 0,
            pagerAssignments: 0,
            educationalEquityAverage: 0,
            violationsCount: 0,
            generationTimeMs: 0
        };

        // Initialize educational profiles
        this.initializeEducationalProfiles();

        // Process existing schedules
        this.processExistingSchedules(existingORSchedules, existingClinicSchedules);
    }

    /**
     * Initialize educational profiles for all residents
     */
    private initializeEducationalProfiles(): void {
        this.residents.forEach(resident => {
            // Initialize caseTypeHours with all case types set to 0
            const caseTypeHours = new Map<CaseType, number>();
            this.CASE_TYPES.forEach(caseType => {
                caseTypeHours.set(caseType, 0);
            });

            const profile: ResidentEducationalProfile = {
                residentId: resident.id,
                surgeonExposureHours: new Map(),
                caseTypeHours,
                weeklyORCount: 0,
                totalORHours: 0,
                clinicHours: 0,
                lastORDate: null,
                educationalDeficits: {
                    surgeonDeficit: 0,
                    caseTypeDeficit: 0,
                    weeklyTargetDeficit: 0
                }
            };

            // Initialize surgeon exposure for all surgeons
            this.surgeons.forEach(surgeon => {
                profile.surgeonExposureHours.set(surgeon.id, 0);
            });

            this.educationalProfiles.set(resident.id, profile);
        });
    }

    /**
     * Process existing OR and clinic schedules
     */
    private processExistingSchedules(
        orSchedules: ORSchedule[],
        clinicSchedules: ClinicSchedule[]
    ): void {
        // Group OR schedules by date
        orSchedules.forEach(schedule => {
            const dateStr = schedule.date.toDate().toDateString();
            if (!this.orSchedules.has(dateStr)) {
                this.orSchedules.set(dateStr, []);
            }
            this.orSchedules.get(dateStr)?.push(schedule);
            
            // Update educational profiles with existing assignments
            if (schedule.assignedResidents) {
                schedule.assignedResidents.forEach(residentId => {
                    const profile = this.educationalProfiles.get(residentId);
                    if (profile) {
                        profile.weeklyORCount++;
                        profile.totalORHours += schedule.estimatedDuration || 4;
                        
                        // Update surgeon exposure
                        const surgeonHours = profile.surgeonExposureHours.get(schedule.surgeonId) || 0;
                        profile.surgeonExposureHours.set(schedule.surgeonId, surgeonHours + (schedule.estimatedDuration || 4));
                        
                        // Update case type hours
                        const caseHours = profile.caseTypeHours.get(schedule.caseType) || 0;
                        profile.caseTypeHours.set(schedule.caseType, caseHours + (schedule.estimatedDuration || 4));
                    }
                });
            }
        });
        
        // Group clinic schedules by date
        clinicSchedules.forEach(schedule => {
            const dateStr = schedule.date.toDate().toDateString();
            if (!this.clinicSchedules.has(dateStr)) {
                this.clinicSchedules.set(dateStr, []);
            }
            this.clinicSchedules.get(dateStr)?.push(schedule);
        });
    }

    /**
     * Generate complete weekly clinical schedule
     */
    public async generateWeeklySchedule(): Promise<ClinicalAssignment[]> {
        const startTime = Date.now();
        
        console.log(`🚀 Starting weekly clinical schedule generation`);
        console.log(`📋 Following prioritized assignment order: OR → Clinic → Float/Pager`);
        
        try {
            // Process each day of the week
            for (let day = 0; day < 7; day++) {
                const currentDate = new Date(this.weekStartDate);
                currentDate.setDate(currentDate.getDate() + day);
                
                console.log(`\n📅 Processing ${currentDate.toDateString()}`);
                
                // HARD CONSTRAINT: No Weekend/Holiday Activities
                if (this.isWeekendOrHoliday(currentDate)) {
                    console.log(`⚠️ Skipping ${currentDate.toDateString()} - Weekend/Holiday protection`);
                    continue;
                }
                
                // Step 1: Process OR assignments (highest priority)
                await this.processORAssignments(currentDate);
                
                // Step 2: Process Clinic assignments (using remaining residents)
                await this.processClinicAssignments(currentDate);
                
                // Step 3: Process Float & Pager duties (lowest priority)
                await this.processFloatAndPagerDuties(currentDate);
            }
            
            // Calculate metrics
            this.calculateMetrics();
            
            this.metrics.generationTimeMs = Date.now() - startTime;
            
            console.log(`\n✅ Weekly schedule generation complete`);
            console.log(`📊 Generated ${this.generatedAssignments.length} total assignments`);
            console.log(`⏱️ Generation time: ${this.metrics.generationTimeMs}ms`);
            console.log(`📈 Average Educational Equity Score: ${this.metrics.educationalEquityAverage.toFixed(2)}`);
            
            if (this.violations.length > 0) {
                console.log(`⚠️ Violations: ${this.violations.length}`);
                this.violations.forEach(v => console.log(`  - ${v}`));
            }
            
            return this.generatedAssignments;
            
        } catch (error) {
            console.error('❌ Error generating weekly schedule:', error);
            throw error;
        }
    }

    /**
     * Process OR assignments for a specific day
     */
    private async processORAssignments(date: Date): Promise<void> {
        const dateStr = date.toDateString();
        const orSchedulesForDay = this.orSchedules.get(dateStr) || [];
        
        if (orSchedulesForDay.length === 0) {
            console.log(`  📋 No OR schedules for ${dateStr}`);
            return;
        }
        
        console.log(`  🔬 Processing ${orSchedulesForDay.length} OR schedules`);
        
        for (const orSchedule of orSchedulesForDay) {
            // HARD CONSTRAINT: Chief Resident Priority
            const chiefAssignment = await this.handleChiefResidentPriority(orSchedule, date);
            if (chiefAssignment) {
                this.generatedAssignments.push(chiefAssignment);
                this.metrics.orAssignments++;
                continue;
            }
            
            // Get available residents for OR
            const availableResidents = this.getAvailableResidentsForOR(date, orSchedule);
            
            if (availableResidents.length === 0) {
                this.violations.push(`No residents available for OR on ${dateStr} with ${orSchedule.surgeonName}`);
                continue;
            }
            
            // Calculate Educational Equity Scores
            const scoredResidents = availableResidents.map(resident => ({
                resident,
                score: this.calculateEducationalEquityScore(resident, orSchedule, date)
            }));
            
            // Sort by score (highest first)
            scoredResidents.sort((a, b) => b.score.total - a.score.total);
            
            // Select primary resident
            const primaryResident = scoredResidents[0].resident;
            const primaryScore = scoredResidents[0].score;
            
            // Create OR assignment for primary resident
            const primaryAssignment = this.createORAssignment(
                primaryResident,
                orSchedule,
                date,
                true,
                primaryScore
            );
            
            // Select team members based on composition rules
            const teamMembers = this.selectTeamMembers(
                primaryResident,
                scoredResidents.slice(1).map(s => s.resident),
                orSchedule
            );
            
            // Add team member IDs to primary assignment
            primaryAssignment.teamMembers = teamMembers.map(r => r.id);
            
            // Create assignments for team members
            const teamAssignments = teamMembers.map(member => 
                this.createORAssignment(
                    member,
                    orSchedule,
                    date,
                    false,
                    this.calculateEducationalEquityScore(member, orSchedule, date)
                )
            );
            
            // Add all assignments
            this.generatedAssignments.push(primaryAssignment);
            this.generatedAssignments.push(...teamAssignments);
            
            // Update educational profiles
            this.updateEducationalProfile(primaryResident.id, orSchedule, true);
            teamMembers.forEach(member => 
                this.updateEducationalProfile(member.id, orSchedule, false)
            );
            
            this.metrics.orAssignments += (1 + teamMembers.length);
            
            console.log(`    ✅ OR Team: ${primaryResident.name} (Primary, Score: ${primaryScore.total.toFixed(2)}) + ${teamMembers.length} assistants`);
        }
    }

    /**
     * Handle Chief Resident priority for OR assignments
     */
    private async handleChiefResidentPriority(
        orSchedule: ORSchedule,
        date: Date
    ): Promise<ORAssignment | null> {
        // Check if any chief resident is pre-assigned to this OR
        const chiefResident = this.residents.find(r => 
            r.isChief && 
            orSchedule.preAssignedResidents?.includes(r.id)
        );
        
        if (!chiefResident) return null;
        
        // Verify chief is available (not on leave or post-call)
        if (!this.isResidentAvailable(chiefResident, date)) {
            this.violations.push(`Chief ${chiefResident.name} pre-assigned but unavailable on ${date.toDateString()}`);
            return null;
        }
        
        console.log(`    👑 Chief priority: ${chiefResident.name} locked for OR`);
        
        return this.createORAssignment(
            chiefResident,
            orSchedule,
            date,
            true,
            this.calculateEducationalEquityScore(chiefResident, orSchedule, date),
            true // chiefOverride flag
        );
    }

    /**
     * Get residents available for OR assignment
     */
    private getAvailableResidentsForOR(date: Date, orSchedule: ORSchedule): Resident[] {
        return this.residents.filter(resident => {
            // Skip if already assigned to another activity today
            if (this.hasAssignmentOnDate(resident.id, date)) return false;
            
            // HARD CONSTRAINT: Vacation & Leave Protection
            if (this.isOnLeave(resident.id, date)) return false;
            
            // HARD CONSTRAINT: Post-Call Protection
            if (this.isPostCall(resident.id, date)) return false;
            
            // Check if resident is qualified for this OR type
            if (!this.isQualifiedForOR(resident, orSchedule)) return false;
            
            return true;
        });
    }

    /**
     * Calculate Educational Equity Score for a resident
     */
    private calculateEducationalEquityScore(
        resident: Resident,
        orSchedule: ORSchedule,
        date: Date
    ): EducationalEquityScore {
        const profile = this.educationalProfiles.get(resident.id);
        if (!profile) {
            return {
                total: 0,
                surgeonExposureDeficit: 0,
                caseTypeDeficit: 0,
                weeklyORTargetDeficit: 0,
                onCallPenalty: 0,
                breakdown: 'No profile found'
            };
        }
        
        // 1. Calculate Surgeon Exposure Deficit
        const avgSurgeonExposure = this.calculateAverageSurgeonExposure(orSchedule.surgeonId);
        const residentSurgeonExposure = profile.surgeonExposureHours.get(orSchedule.surgeonId) || 0;
        const surgeonDeficit = Math.max(0, avgSurgeonExposure - residentSurgeonExposure);
        
        // 2. Calculate Case Type Deficit
        const targetCaseHours = this.getTargetCaseTypeHours(resident.pgyLevel, orSchedule.caseType);
        const currentCaseHours = profile.caseTypeHours.get(orSchedule.caseType) || 0;
        const caseTypeDeficit = Math.max(0, targetCaseHours - currentCaseHours);
        
        // 3. Calculate Weekly OR Target Deficit
        const weeklyTarget = this.WEEKLY_OR_TARGETS[resident.pgyLevel] || 3;
        const weeklyDeficit = Math.max(0, weeklyTarget - profile.weeklyORCount);
        
        // 4. On-Call Penalty (residents on call can still be assigned but with penalty)
        const isOnCall = this.isOnCall(resident.id, date);
        const onCallPenalty = isOnCall ? 10 : 0; // Penalty value
        
        // Calculate weighted total score
        const total = 
            (surgeonDeficit * this.SCORE_WEIGHTS.surgeonExposure) +
            (caseTypeDeficit * this.SCORE_WEIGHTS.caseType) +
            (weeklyDeficit * 10 * this.SCORE_WEIGHTS.weeklyTarget) + // Scale weekly deficit
            (onCallPenalty * this.SCORE_WEIGHTS.onCallPenalty);
        
        return {
            total,
            surgeonExposureDeficit: surgeonDeficit,
            caseTypeDeficit: caseTypeDeficit,
            weeklyORTargetDeficit: weeklyDeficit,
            onCallPenalty: onCallPenalty,
            breakdown: `Surgeon: ${surgeonDeficit.toFixed(1)}h, Case: ${caseTypeDeficit.toFixed(1)}h, Weekly: ${weeklyDeficit}, OnCall: ${onCallPenalty}`
        };
    }

    /**
     * Select team members based on composition rules
     */
    private selectTeamMembers(
        primaryResident: Resident,
        availableResidents: Resident[],
        orSchedule: ORSchedule
    ): Resident[] {
        const teamMembers: Resident[] = [];
        const requiredTeamSize = orSchedule.requiredResidents || 2;
        
        // Already have primary, need (requiredTeamSize - 1) more
        const neededMembers = requiredTeamSize - 1;
        
        // Filter by PGY level difference rule
        const eligibleMembers = availableResidents.filter(resident => {
            const pgyDifference = Math.abs(resident.pgyLevel - primaryResident.pgyLevel);
            
            // Team Composition Rule: At least 2-year PGY difference
            if (pgyDifference < 2 && resident.pgyLevel !== 1) {
                return false;
            }
            
            return true;
        });
        
        // Sort by educational equity score for this OR
        eligibleMembers.sort((a, b) => {
            const scoreA = this.calculateEducationalEquityScore(a, orSchedule, orSchedule.date.toDate());
            const scoreB = this.calculateEducationalEquityScore(b, orSchedule, orSchedule.date.toDate());
            return scoreB.total - scoreA.total;
        });
        
        // Select team members
        for (let i = 0; i < Math.min(neededMembers, eligibleMembers.length); i++) {
            teamMembers.push(eligibleMembers[i]);
            
            // PGY-1 Exception: Can have 3 residents if one is PGY-1
            if (teamMembers.length === 2 && 
                requiredTeamSize === 3 && 
                !teamMembers.some(m => m.pgyLevel === 1) &&
                !primaryResident.pgyLevel !== 1) {
                // Need a PGY-1 for third member
                const pgy1 = eligibleMembers.find(r => 
                    r.pgyLevel === 1 && !teamMembers.includes(r)
                );
                if (pgy1) {
                    teamMembers.push(pgy1);
                }
                break;
            }
        }
        
        return teamMembers;
    }

    /**
     * Process clinic assignments for a specific day
     */
    private async processClinicAssignments(date: Date): Promise<void> {
        const dateStr = date.toDateString();
        const clinicSchedulesForDay = this.clinicSchedules.get(dateStr) || [];
        
        if (clinicSchedulesForDay.length === 0) {
            console.log(`  📋 No clinic schedules for ${dateStr}`);
            return;
        }
        
        console.log(`  🏥 Processing ${clinicSchedulesForDay.length} clinic schedules`);
        
        for (const clinicSchedule of clinicSchedulesForDay) {
            // Determine required staffing based on patient volume
            const requiredStaff = this.calculateRequiredClinicStaff(clinicSchedule.expectedPatients);
            
            // Get available residents (excluding those already in OR)
            const availableResidents = this.getAvailableResidentsForClinic(date);
            
            if (availableResidents.length < requiredStaff) {
                this.violations.push(`Insufficient residents for clinic on ${dateStr} (need ${requiredStaff}, have ${availableResidents.length})`);
            }
            
            // Prioritize assignment order
            const prioritizedResidents = this.prioritizeForClinic(availableResidents);
            
            // Select required number of residents
            const selectedResidents = prioritizedResidents.slice(0, requiredStaff);
            
            // Create clinic assignment
            const clinicAssignment = this.createClinicAssignment(
                clinicSchedule,
                selectedResidents,
                date
            );
            
            this.generatedAssignments.push(clinicAssignment);
            this.metrics.clinicAssignments++;
            
            // Update profiles
            selectedResidents.forEach(resident => {
                const profile = this.educationalProfiles.get(resident.id);
                if (profile) {
                    profile.clinicHours += 4; // Assume 4-hour clinic session
                }
            });
            
            console.log(`    ✅ Clinic: ${selectedResidents.map(r => r.name).join(', ')} (${clinicSchedule.expectedPatients} patients)`);
        }
    }

    /**
     * Calculate required clinic staffing based on patient volume
     */
    private calculateRequiredClinicStaff(expectedPatients: number): number {
        const rule = this.CLINIC_STAFFING_RULES.find(r => expectedPatients >= r.minPatients);
        return rule?.requiredStaff || 1;
    }

    /**
     * Get residents available for clinic assignment
     */
    private getAvailableResidentsForClinic(date: Date): Resident[] {
        return this.residents.filter(resident => {
            // Skip if already assigned to OR or other activity today
            if (this.hasAssignmentOnDate(resident.id, date)) return false;
            
            // Apply same hard constraints
            if (this.isOnLeave(resident.id, date)) return false;
            if (this.isPostCall(resident.id, date)) return false;
            
            return true;
        });
    }

    /**
     * Prioritize residents for clinic assignment
     */
    private prioritizeForClinic(residents: Resident[]): Resident[] {
        return residents.sort((a, b) => {
            // Priority 1: Non-neurosurgery residents
            if (a.service !== 'Neurosurgery' && b.service === 'Neurosurgery') return -1;
            if (a.service === 'Neurosurgery' && b.service !== 'Neurosurgery') return 1;
            
            // Priority 2: Junior residents (lower PGY)
            if (a.pgyLevel !== b.pgyLevel) return a.pgyLevel - b.pgyLevel;
            
            // Priority 3: Less clinic exposure
            const profileA = this.educationalProfiles.get(a.id);
            const profileB = this.educationalProfiles.get(b.id);
            
            return (profileA?.clinicHours || 0) - (profileB?.clinicHours || 0);
        });
    }

    /**
     * Process float and pager duties
     */
    private async processFloatAndPagerDuties(date: Date): Promise<void> {
        // Get remaining available residents
        const availableResidents = this.residents.filter(resident => {
            if (this.hasAssignmentOnDate(resident.id, date)) return false;
            if (this.isOnLeave(resident.id, date)) return false;
            if (this.isPostCall(resident.id, date)) return false;
            return true;
        });
        
        if (availableResidents.length === 0) return;
        
        // Assign float duty (if needed)
        if (this.needsFloatCoverage(date) && availableResidents.length > 0) {
            const floatResident = availableResidents[0];
            const floatAssignment = this.createFloatAssignment(floatResident, date);
            this.generatedAssignments.push(floatAssignment);
            this.metrics.floatAssignments++;
            
            console.log(`    ✅ Float: ${floatResident.name}`);
            
            // Remove from available pool
            availableResidents.shift();
        }
        
        // Assign pager duty (if needed)
        if (this.needsPagerCoverage(date) && availableResidents.length > 0) {
            const pagerResident = availableResidents[0];
            const pagerAssignment = this.createPagerAssignment(pagerResident, date);
            this.generatedAssignments.push(pagerAssignment);
            this.metrics.pagerAssignments++;
            
            console.log(`    ✅ Pager: ${pagerResident.name}`);
        }
    }

    /**
     * Create OR assignment
     */
    private createORAssignment(
        resident: Resident,
        orSchedule: ORSchedule,
        date: Date,
        isPrimary: boolean,
        educationalScore: EducationalEquityScore,
        chiefOverride: boolean = false
    ): ORAssignment {
        return {
            id: `or-${uuidv4()}`,
            residentId: resident.id,
            residentName: resident.name,
            date: admin.firestore.Timestamp.fromDate(date),
            type: 'OR',
            location: orSchedule.location || 'Main OR',
            startTime: orSchedule.startTime,
            endTime: orSchedule.endTime,
            surgeonId: orSchedule.surgeonId,
            surgeonName: orSchedule.surgeonName,
            caseType: orSchedule.caseType,
            isPrimary,
            teamMembers: [],
            estimatedDuration: orSchedule.estimatedDuration || 4,
            educationalScore: educationalScore.total,
            chiefOverride,
            status: 'Scheduled',
            createdAt: admin.firestore.Timestamp.now(),
            updatedAt: admin.firestore.Timestamp.now()
        } as ORAssignment;
    }

    /**
     * Create clinic assignment
     */
    private createClinicAssignment(
        clinicSchedule: ClinicSchedule,
        residents: Resident[],
        date: Date
    ): ClinicAssignment {
        return {
            id: `clinic-${uuidv4()}`,
            residentId: residents[0].id, // Primary resident
            residentName: residents.map(r => r.name).join(', '),
            date: admin.firestore.Timestamp.fromDate(date),
            type: 'Clinic',
            location: clinicSchedule.location || 'Outpatient Clinic',
            startTime: clinicSchedule.startTime,
            endTime: clinicSchedule.endTime,
            clinicType: clinicSchedule.clinicType,
            expectedPatients: clinicSchedule.expectedPatients,
            requiredStaffing: residents.length,
            assignedResidents: residents.map(r => r.id),
            status: 'Scheduled',
            createdAt: admin.firestore.Timestamp.now(),
            updatedAt: admin.firestore.Timestamp.now()
        } as ClinicAssignment;
    }

    /**
     * Create float assignment
     */
    private createFloatAssignment(resident: Resident, date: Date): ClinicalAssignment {
        return {
            id: `float-${uuidv4()}`,
            residentId: resident.id,
            residentName: resident.name,
            date: admin.firestore.Timestamp.fromDate(date),
            type: 'Float',
            location: 'Hospital',
            status: 'Scheduled',
            createdAt: admin.firestore.Timestamp.now(),
            updatedAt: admin.firestore.Timestamp.now()
        };
    }

    /**
     * Create pager assignment
     */
    private createPagerAssignment(resident: Resident, date: Date): ClinicalAssignment {
        return {
            id: `pager-${uuidv4()}`,
            residentId: resident.id,
            residentName: resident.name,
            date: admin.firestore.Timestamp.fromDate(date),
            type: 'Pager',
            location: 'Hospital',
            status: 'Scheduled',
            createdAt: admin.firestore.Timestamp.now(),
            updatedAt: admin.firestore.Timestamp.now()
        };
    }

    /**
     * Update educational profile after OR assignment
     */
    private updateEducationalProfile(
        residentId: string,
        orSchedule: ORSchedule,
        isPrimary: boolean
    ): void {
        const profile = this.educationalProfiles.get(residentId);
        if (!profile) return;
        
        const duration = orSchedule.estimatedDuration || 4;
        
        // Update surgeon exposure
        const currentSurgeonHours = profile.surgeonExposureHours.get(orSchedule.surgeonId) || 0;
        profile.surgeonExposureHours.set(orSchedule.surgeonId, currentSurgeonHours + duration);
        
        // Update case type hours
        const currentCaseHours = profile.caseTypeHours.get(orSchedule.caseType) || 0;
        profile.caseTypeHours.set(orSchedule.caseType, currentCaseHours + duration);
        
        // Update OR counts
        profile.weeklyORCount++;
        profile.totalORHours += duration;
        profile.lastORDate = orSchedule.date.toDate();
        
        // Recalculate deficits
        this.recalculateDeficits(profile);
    }

    /**
     * Recalculate educational deficits
     */
    private recalculateDeficits(profile: ResidentEducationalProfile): void {
        // This would be more complex in production, considering curriculum requirements
        profile.educationalDeficits.surgeonDeficit = 0;
        profile.educationalDeficits.caseTypeDeficit = 0;
        profile.educationalDeficits.weeklyTargetDeficit = 0;
    }

    /**
     * Check if date is weekend or holiday
     */
    private isWeekendOrHoliday(date: Date): boolean {
        const dayOfWeek = date.getDay();
        
        // Weekend check
        if (dayOfWeek === 0 || dayOfWeek === 6) return true;
        
        // Holiday check
        return this.academicYear.holidays.some(holiday => {
            const holidayDate = holiday.date.toDate();
            return this.isSameDay(date, holidayDate);
        });
    }

    /**
     * Check if resident is available
     */
    private isResidentAvailable(resident: Resident, date: Date): boolean {
        return !this.isOnLeave(resident.id, date) && 
               !this.isPostCall(resident.id, date);
    }

    /**
     * Check if resident has assignment on date
     */
    private hasAssignmentOnDate(residentId: string, date: Date): boolean {
        return this.generatedAssignments.some(assignment =>
            assignment.residentId === residentId &&
            this.isSameDay(assignment.date.toDate(), date)
        );
    }

    /**
     * Check if resident is on leave
     */
    private isOnLeave(residentId: string, date: Date): boolean {
        return this.approvedLeave.some(leave => {
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
        const previousDay = new Date(date);
        previousDay.setDate(date.getDate() - 1);
        
        return this.callSchedule.some(call => {
            if (call.residentId !== residentId) return false;
            if (!this.isSameDay(call.date.toDate(), previousDay)) return false;
            
            // Post-call protection applies for 24h, night, and weekend calls
            return call.type === '24h' || 
                   call.type === 'Night' || 
                   call.type === 'Weekend' ||
                   call.type === 'Holiday';
        });
    }

    /**
     * Check if resident is on call
     */
    private isOnCall(residentId: string, date: Date): boolean {
        return this.callSchedule.some(call =>
            call.residentId === residentId &&
            this.isSameDay(call.date.toDate(), date) &&
            call.type !== 'PostCall'
        );
    }

    /**
     * Check if resident is qualified for OR
     */
    private isQualifiedForOR(resident: Resident, orSchedule: ORSchedule): boolean {
        // Check minimum PGY level for case complexity
        if (orSchedule.minimumPGY && resident.pgyLevel < orSchedule.minimumPGY) {
            return false;
        }
        
        // Check if resident's service matches OR requirements
        if (orSchedule.requiredService && resident.service !== orSchedule.requiredService) {
            return false;
        }
        
        return true;
    }

    /**
     * Calculate average surgeon exposure across all residents
     */
    private calculateAverageSurgeonExposure(surgeonId: string): number {
        let totalHours = 0;
        let residentCount = 0;
        
        this.educationalProfiles.forEach(profile => {
            const hours = profile.surgeonExposureHours.get(surgeonId) || 0;
            totalHours += hours;
            residentCount++;
        });
        
        return residentCount > 0 ? totalHours / residentCount : 0;
    }

    /**
     * Get target case type hours based on PGY level
     */
    private getTargetCaseTypeHours(pgyLevel: number, caseType: CaseType): number {
        // These would be defined based on curriculum requirements
        const targets: Record<number, Record<CaseType, number>> = {
            1: { 'Spine': 20, 'Cranial': 30, 'Pediatric': 10, 'Functional': 5, 'Vascular': 15 },
            2: { 'Spine': 40, 'Cranial': 60, 'Pediatric': 20, 'Functional': 10, 'Vascular': 30 },
            3: { 'Spine': 80, 'Cranial': 100, 'Pediatric': 30, 'Functional': 20, 'Vascular': 50 },
            4: { 'Spine': 120, 'Cranial': 150, 'Pediatric': 40, 'Functional': 30, 'Vascular': 70 },
            5: { 'Spine': 160, 'Cranial': 200, 'Pediatric': 50, 'Functional': 40, 'Vascular': 90 },
            6: { 'Spine': 200, 'Cranial': 250, 'Pediatric': 60, 'Functional': 50, 'Vascular': 110 },
            7: { 'Spine': 240, 'Cranial': 300, 'Pediatric': 70, 'Functional': 60, 'Vascular': 130 }
        };
        
        return targets[pgyLevel]?.[caseType] || 50;
    }

    /**
     * Check if float coverage is needed
     */
    private needsFloatCoverage(date: Date): boolean {
        // Custom logic to determine if float is needed
        // For now, assume float is needed on weekdays
        const dayOfWeek = date.getDay();
        return dayOfWeek >= 1 && dayOfWeek <= 5;
    }

    /**
     * Check if pager coverage is needed
     */
    private needsPagerCoverage(date: Date): boolean {
        // Custom logic to determine if pager is needed
        // For now, assume pager is needed every day
        return true;
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
     * Calculate metrics for the generated schedule
     */
    private calculateMetrics(): void {
        // Calculate average educational equity score
        let totalScore = 0;
        let scoreCount = 0;
        
        this.generatedAssignments.forEach(assignment => {
            if (assignment.type === 'OR') {
                const orAssignment = assignment as ORAssignment;
                if (orAssignment.educationalScore) {
                    totalScore += orAssignment.educationalScore;
                    scoreCount++;
                }
            }
        });
        
        this.metrics.educationalEquityAverage = scoreCount > 0 ? totalScore / scoreCount : 0;
        this.metrics.violationsCount = this.violations.length;
    }

    /**
     * Get schedule analytics
     */
    public getAnalytics(): any {
        return {
            weekRange: `${this.weekStartDate.toDateString()} - ${this.weekEndDate.toDateString()}`,
            metrics: this.metrics,
            educationalProfiles: Array.from(this.educationalProfiles.values()).map(profile => ({
                residentId: profile.residentId,
                weeklyORCount: profile.weeklyORCount,
                totalORHours: profile.totalORHours,
                clinicHours: profile.clinicHours,
                surgeonExposure: Array.from(profile.surgeonExposureHours.entries()),
                caseTypeHours: Array.from(profile.caseTypeHours.entries())
            })),
            violations: this.violations,
            assignmentBreakdown: {
                OR: this.metrics.orAssignments,
                Clinic: this.metrics.clinicAssignments,
                Float: this.metrics.floatAssignments,
                Pager: this.metrics.pagerAssignments,
                Total: this.generatedAssignments.length
            }
        };
    }

    /**
     * Export weekly schedule
     */
    public exportSchedule(format: 'json' | 'csv' = 'json'): string {
        if (format === 'csv') {
            return this.exportToCSV();
        }
        return this.exportToJSON();
    }

    /**
     * Export to JSON
     */
    private exportToJSON(): string {
        return JSON.stringify({
            metadata: {
                weekStart: this.weekStartDate.toISOString(),
                weekEnd: this.weekEndDate.toISOString(),
                generated: new Date().toISOString(),
                generatedBy: 'ramihatou97'
            },
            assignments: this.generatedAssignments,
            analytics: this.getAnalytics()
        }, null, 2);
    }

    /**
     * Export to CSV
     */
    private exportToCSV(): string {
        const headers = ['Date', 'Day', 'Resident', 'PGY', 'Type', 'Location', 'Details'];
        const rows = [headers.join(',')];
        
        this.generatedAssignments
            .sort((a, b) => a.date.toDate().getTime() - b.date.toDate().getTime())
            .forEach(assignment => {
                const date = assignment.date.toDate();
                const resident = this.residents.find(r => r.id === assignment.residentId);
                const dayName = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][date.getDay()];
                
                let details = '';
                if (assignment.type === 'OR') {
                    const orAssignment = assignment as ORAssignment;
                    details = `${orAssignment.surgeonName} - ${orAssignment.caseType} ${orAssignment.isPrimary ? '(Primary)' : '(Assistant)'}`;
                } else if (assignment.type === 'Clinic') {
                    const clinicAssignment = assignment as ClinicAssignment;
                    details = `${clinicAssignment.expectedPatients} patients`;
                }
                
                rows.push([
                    date.toLocaleDateString(),
                    dayName,
                    assignment.residentName,
                    resident?.pgyLevel || '',
                    assignment.type,
                    assignment.location || '',
                    details
                ].join(','));
            });
        
        return rows.join('\n');
    }
}

// Export the scheduler
export default WeeklyClinicalScheduler;