    // --- AI Optimization: Simulated Annealing (placeholder) ---
    /**
     * Optimize the yearly schedule using simulated annealing
     * This is a placeholder for advanced AI optimization
     */
    public async optimizeSchedule(): Promise<void> {
        console.log('🤖 Starting AI optimization (simulated annealing)...');
        // Simulated annealing parameters
        let temperature = 1.0;
        const minTemperature = 0.01;
        const coolingRate = 0.95;
        const maxIterations = 200;
        let bestAssignments = new Map<string, YearlyRotationAssignment[]>(
            Array.from(this.assignments.entries()).map(([k, v]) => [k, v.map(a => ({...a}))])
        );
        let bestViolations = this.violations.length;

        for (let iter = 0; iter < maxIterations && temperature > minTemperature; iter++) {
            // Pick two random residents and a random block
            const residentIds = Array.from(this.assignments.keys());
            const r1 = residentIds[Math.floor(Math.random() * residentIds.length)];
            const r2 = residentIds[Math.floor(Math.random() * residentIds.length)];
            if (r1 === r2) continue;
            const block = Math.floor(Math.random() * this.TOTAL_BLOCKS) + 1;

            // Swap assignments for this block
            const a1 = (this.assignments.get(r1) || []).find(a => a.blockNumber === block);
            const a2 = (this.assignments.get(r2) || []).find(a => a.blockNumber === block);
            if (!a1 || !a2) continue;

            // Swap
            a1.residentId = r2;
            a2.residentId = r1;
            // Re-assign in map
            let arr1 = this.assignments.get(r1) || [];
            let arr2 = this.assignments.get(r2) || [];
            arr1 = arr1.map(a => a.blockNumber === block ? a2 : a);
            arr2 = arr2.map(a => a.blockNumber === block ? a1 : a);
            this.assignments.set(r1, arr1);
            this.assignments.set(r2, arr2);

            // Recalculate violations
            this.violations = [];
            this.blockCoverage.forEach((_, blockNumber) => this.validateBlockCoverage(blockNumber));
            this.validateSchedule();
            const newViolations = this.violations.length;

            // Accept if better or with probability
            const delta = newViolations - bestViolations;
            if (delta < 0 || Math.exp(-delta / temperature) > Math.random()) {
                if (newViolations < bestViolations) {
                    bestViolations = newViolations;
                    bestAssignments = new Map<string, YearlyRotationAssignment[]>(
                        Array.from(this.assignments.entries()).map(([k, v]) => [k, v.map(a => ({...a}))])
                    );
                }
            } else {
                // Revert swap
                a1.residentId = r1;
                a2.residentId = r2;
                arr1 = arr1.map(a => a.blockNumber === block ? a1 : a);
                arr2 = arr2.map(a => a.blockNumber === block ? a2 : a);
                this.assignments.set(r1, arr1);
                this.assignments.set(r2, arr2);
            }
            temperature *= coolingRate;
        }
        // Restore best found
        this.assignments = new Map<string, YearlyRotationAssignment[]>(
            Array.from(bestAssignments.entries()).map(([k, v]) => [k, v.map(a => ({...a}))])
        );
        this.violations = [];
        this.blockCoverage.forEach((_, blockNumber) => this.validateBlockCoverage(blockNumber));
        this.validateSchedule();
        console.log('✅ AI optimization complete. Best violations:', bestViolations);
    }

    // --- RCPSC Requirements Validation ---
    /**
     * Validate schedule against RCPSC requirements
     * Returns true if compliant, false otherwise
     */
    public validateRCPSCCompliance(): boolean {
        console.log('🔎 Validating RCPSC requirements...');
        let compliant = true;
        // Check each block for coverage requirements
        this.blockCoverage.forEach((coverage, blockNumber) => {
            if (!coverage.meetsRequirements) {
                console.warn(`Block ${blockNumber} is not RCPSC compliant: ${coverage.deficits.join('; ')}`);
                compliant = false;
            }
        });
        // Additional RCPSC-specific checks can be added here
        // e.g., minimum number of core rotations, exam leave, etc.
        if (compliant) {
            console.log('✅ Schedule is RCPSC compliant');
        } else {
            console.log('❌ Schedule is NOT RCPSC compliant');
        }
        return compliant;
    }
import { 
    Resident, 
    AppConfiguration, 
    AcademicYear,
    RotationBlock,
    RotationType,
    ExternalRotator,
    HolidayPointsEntry,
    TeamAssignment,
    CoverageRequirement,
    RotationAssignment,
    MandatoryRotation,
    ExamLeaveRequirement
} from '../../../shared/types';
import * as admin from 'firebase-admin';
import { v4 as uuidv4 } from 'uuid';

/**
 * External rotator types with availability
 */
interface ExternalRotatorProfile {
    residentId: string;
    residentName: string;
    service: string;
    type: 'Inflexible' | 'Flexible';
    assignedBlocks: number[]; // For inflexible rotators
    availableBlocks: number[]; // For flexible rotators
    pgyLevel: number;
}

/**
 * Block coverage analysis
 */
interface BlockCoverage {
    blockNumber: number;
    totalResidents: number;
    neurosurgeryResidents: number;
    seniorResidents: number; // PGY-4+
    externalRotators: number;
    meetsRequirements: boolean;
    deficits: string[];
    projectedCoverage: number; // For planning
    redTeamCount: number;
    blueTeamCount: number;
}

/**
 * Rotation assignment with metadata
 */
interface YearlyRotationAssignment extends RotationAssignment {
    blockNumber: number;
    blockStartDate: Date;
    blockEndDate: Date;
    rotationType: RotationType;
    isMandatory: boolean;
    isExamLeave: boolean;
    isHolidayLeave: boolean;
    isConsecutive: boolean;
    consecutiveBlockIds: string[];
    teamAssignment?: 'Red' | 'Blue';
    phase: number; // Which phase created this assignment
}

/**
 * Schedule generation metrics
 */
interface YearlyScheduleMetrics {
    totalAssignments: number;
    phaseDurations: Map<number, number>;
    coverageViolations: number;
    externalRotatorsPlaced: number;
    flexibleRotatorsOptimized: number;
    holidayAssignments: number;
    teamBalanceScore: number;
    generationTimeMs: number;
    consecutiveBlocksCreated: number;
}

/**
 * Holiday points tracking
 */
interface HolidayPointsTracker {
    residentId: string;
    residentName: string;
    totalPoints: number;
    christmasEligible: boolean;
    newYearEligible: boolean;
    awarded: 'Christmas' | 'NewYear' | null;
}

/**
 * Complete Yearly Rotation Scheduler - Production Version
 * 
 * Implements all features:
 * - Hard Constraints: Mandatory leave blocks, Consecutive blocks, Inflexible rotations
 * - Intelligent Assignment: Phased generation, Customizable coverage rules, Holiday points
 * - Advanced Features: External rotator integration (both types), Team balancing
 * 
 * Created: 2025-08-16 06:17:26 UTC by ramihatou97
 * Version: 1.0.0 (Production Release)
 */
export class YearlyRotationScheduler {
    private residents: Resident[];
    private config: AppConfiguration;
    private academicYear: AcademicYear;
    private rotationBlocks: RotationBlock[];
    private externalRotators: ExternalRotatorProfile[];
    private holidayPoints: Map<string, HolidayPointsTracker>;
    private assignments: Map<string, YearlyRotationAssignment[]>; // residentId -> assignments
    private blockCoverage: Map<number, BlockCoverage>;
    private coverageRequirements: CoverageRequirement;
    private mandatoryRotations: MandatoryRotation[];
    private examLeaveRequirements: ExamLeaveRequirement[];
    private metrics: YearlyScheduleMetrics;
    private violations: string[] = [];
    
    // Define the 13 blocks in the academic year
    private readonly TOTAL_BLOCKS = 13;
    private readonly BLOCK_DURATION_WEEKS = 4;
    
    // Phase definitions
    private readonly PHASES = {
        0: 'External Rotators',
        1: 'Inflexible Off-Service',
        2: 'Mandatory Exam Leave',
        3: 'Holiday Leave Assignment',
        4: 'Core Neurosurgery Rotations',
        5: 'Flexible Rotations & Electives',
        6: 'Team Balancing'
    };
    
    // Default coverage requirements (customizable)
    private readonly DEFAULT_COVERAGE_REQUIREMENTS: CoverageRequirement = {
        minTotalResidents: 4,
        minNeurosurgeryResidents: 3,
        minSeniorResidents: 1, // PGY-4+
        maxResidentsOnLeave: 2,
        requireChiefPresence: false
    };

    constructor(
        residents: Resident[],
        config: AppConfiguration,
        academicYear: AcademicYear,
        externalRotators: ExternalRotator[],
        holidayPointsData: HolidayPointsEntry[],
        mandatoryRotations: MandatoryRotation[],
        examLeaveRequirements: ExamLeaveRequirement[],
        customCoverageRequirements?: Partial<CoverageRequirement>
    ) {
        this.residents = residents;
        this.config = config;
        this.academicYear = academicYear;
        this.mandatoryRotations = mandatoryRotations;
        this.examLeaveRequirements = examLeaveRequirements;
        
        // Merge custom requirements with defaults
        this.coverageRequirements = {
            ...this.DEFAULT_COVERAGE_REQUIREMENTS,
            ...customCoverageRequirements
        };
        
        console.log(`🗓️ Initializing Yearly Rotation Scheduler`);
        console.log(`📅 Academic Year: ${academicYear.name} (${academicYear.startDate.toDate().toDateString()} - ${academicYear.endDate.toDate().toDateString()})`);
        console.log(`👥 Total Residents: ${residents.length}`);
        console.log(`🌐 External Rotators: ${externalRotators.length}`);
        console.log(`📝 Mandatory Rotations: ${mandatoryRotations.length}`);
        console.log(`📚 Exam Leave Requirements: ${examLeaveRequirements.length}`);
        console.log(`👤 User: ramihatou97 | Time: 2025-08-16 06:17:26 UTC`);
        
        // Initialize data structures
        this.rotationBlocks = [];
        this.externalRotators = [];
        this.holidayPoints = new Map();
        this.assignments = new Map();
        this.blockCoverage = new Map();
        
        this.metrics = {
            totalAssignments: 0,
            phaseDurations: new Map(),
            coverageViolations: 0,
            externalRotatorsPlaced: 0,
            flexibleRotatorsOptimized: 0,
            holidayAssignments: 0,
            teamBalanceScore: 0,
            generationTimeMs: 0,
            consecutiveBlocksCreated: 0
        };
        
        // Initialize rotation blocks
        this.initializeRotationBlocks();
        
        // Process external rotators
        this.processExternalRotators(externalRotators);
        
        // Initialize holiday points
        this.initializeHolidayPoints(holidayPointsData);
        
        // Initialize empty assignments for all residents
        this.residents.forEach(resident => {
            this.assignments.set(resident.id, []);
        });
    }

    /**
     * Initialize the 13 rotation blocks for the academic year
     */
    private initializeRotationBlocks(): void {
        const startDate = this.academicYear.startDate.toDate();
        
        for (let i = 0; i < this.TOTAL_BLOCKS; i++) {
            const blockStart = new Date(startDate);
            blockStart.setDate(startDate.getDate() + (i * this.BLOCK_DURATION_WEEKS * 7));
            
            const blockEnd = new Date(blockStart);
            blockEnd.setDate(blockStart.getDate() + (this.BLOCK_DURATION_WEEKS * 7) - 1);
            
            const block: RotationBlock = {
                id: `block-${this.academicYear.id}-${i + 1}`,
                blockNumber: i + 1,
                startDate: admin.firestore.Timestamp.fromDate(blockStart),
                endDate: admin.firestore.Timestamp.fromDate(blockEnd),
                academicYearId: this.academicYear.id,
                name: `Block ${i + 1}`,
                isHolidayBlock: this.isHolidayBlock(i + 1)
            };
            
            this.rotationBlocks.push(block);
            
            // Initialize coverage tracking
            this.blockCoverage.set(i + 1, {
                blockNumber: i + 1,
                totalResidents: 0,
                neurosurgeryResidents: 0,
                seniorResidents: 0,
                externalRotators: 0,
                meetsRequirements: false,
                deficits: [],
                projectedCoverage: this.residents.filter(r => r.service === 'Neurosurgery').length,
                redTeamCount: 0,
                blueTeamCount: 0
            });
        }
        
        console.log(`📊 Initialized ${this.TOTAL_BLOCKS} rotation blocks`);
    }

    /**
     * Check if a block is a holiday block (Christmas or New Year)
     */
    private isHolidayBlock(blockNumber: number): boolean {
        // Blocks 6-7 typically cover December/January
        // This should be configured based on actual calendar
        const block = this.rotationBlocks[blockNumber - 1];
        if (!block) return false;
        
        const blockStart = block.startDate.toDate();
        const blockEnd = block.endDate.toDate();
        
        // Check if block contains December 25 (Christmas) or January 1 (New Year)
        const christmas = new Date(blockStart.getFullYear(), 11, 25); // December 25
        const newYear = new Date(blockStart.getFullYear() + 1, 0, 1); // January 1
        
        return (blockStart <= christmas && christmas <= blockEnd) ||
               (blockStart <= newYear && newYear <= blockEnd);
    }

    /**
     * Process external rotators and categorize them
     */
    private processExternalRotators(externalRotators: ExternalRotator[]): void {
        externalRotators.forEach(rotator => {
            const profile: ExternalRotatorProfile = {
                residentId: rotator.residentId,
                residentName: rotator.residentName,
                service: rotator.service,
                type: rotator.type,
                assignedBlocks: rotator.assignedBlocks || [],
                availableBlocks: rotator.availableBlocks || [],
                pgyLevel: rotator.pgyLevel
            };
            
            this.externalRotators.push(profile);
            
            // Add to residents list if not already present
            if (!this.residents.find(r => r.id === rotator.residentId)) {
                this.residents.push({
                    id: rotator.residentId,
                    name: rotator.residentName,
                    pgyLevel: rotator.pgyLevel,
                    service: rotator.service,
                    isChief: false,
                    callExempt: false,
                    email: '',
                    phoneNumber: ''
                } as Resident);
            }
        });
        
        const inflexibleCount = this.externalRotators.filter(r => r.type === 'Inflexible').length;
        const flexibleCount = this.externalRotators.filter(r => r.type === 'Flexible').length;
        
        console.log(`🔄 Processed ${inflexibleCount} inflexible and ${flexibleCount} flexible external rotators`);
    }

    /**
     * Initialize holiday points tracking
     */
    private initializeHolidayPoints(holidayPointsData: HolidayPointsEntry[]): void {
        holidayPointsData.forEach(entry => {
            const resident = this.residents.find(r => r.id === entry.residentId);
            this.holidayPoints.set(entry.residentId, {
                residentId: entry.residentId,
                residentName: resident?.name || 'Unknown',
                totalPoints: entry.totalPoints,
                christmasEligible: entry.christmasEligible !== false,
                newYearEligible: entry.newYearEligible !== false,
                awarded: null
            });
        });
        
        console.log(`🎄 Initialized holiday points for ${holidayPointsData.length} residents`);
    }

    /**
     * Generate the complete yearly rotation schedule
     */
    public async generateYearlySchedule(): Promise<YearlyRotationAssignment[]> {
        const startTime = Date.now();
        
        console.log(`\n🚀 Starting Yearly Rotation Schedule Generation`);
        console.log(`📋 Using phased approach: ${Object.values(this.PHASES).join(' → ')}`);
        
        try {
            // Phase 0: Place External Rotators (Inflexible first)
            await this.executePhase0_ExternalRotators();
            
            // Phase 1: Place Inflexible Off-Service Rotations
            await this.executePhase1_InflexibleOffService();
            
            // Phase 2: Place Mandatory Exam Leave
            await this.executePhase2_MandatoryExamLeave();
            
            // Phase 3: Assign Holiday Leave via Points System
            await this.executePhase3_HolidayLeave();
            
            // Phase 4: Fill Core Neurosurgery Rotations
            await this.executePhase4_CoreRotations();
            
            // Phase 5: Place Flexible Rotations and Electives
            await this.executePhase5_FlexibleRotations();
            
            // Phase 6: Balance Teams
            await this.executePhase6_TeamBalancing();
            
            // Validate final schedule
            this.validateSchedule();
            
            // Calculate metrics
            this.metrics.generationTimeMs = Date.now() - startTime;
            this.metrics.totalAssignments = this.getAllAssignments().length;
            
            console.log(`\n✅ Schedule generation complete`);
            console.log(`📊 Total assignments: ${this.metrics.totalAssignments}`);
            console.log(`⏱️ Generation time: ${this.metrics.generationTimeMs}ms`);
            console.log(`⚠️ Coverage violations: ${this.metrics.coverageViolations}`);
            console.log(`🔗 Consecutive blocks created: ${this.metrics.consecutiveBlocksCreated}`);
            
            if (this.violations.length > 0) {
                console.log(`\n❌ Violations detected:`);
                this.violations.forEach(v => console.log(`  - ${v}`));
            }
            
            return this.getAllAssignments();
            
        } catch (error) {
            console.error('❌ Error generating yearly schedule:', error);
            throw error;
        }
    }

    /**
     * Phase 0: Place External Rotators
     */
    private async executePhase0_ExternalRotators(): Promise<void> {
        const phaseStart = Date.now();
        console.log(`\n🔄 Phase 0: Placing External Rotators`);
        
        // First, place inflexible external rotators (hard constraints)
        const inflexibleRotators = this.externalRotators.filter(r => r.type === 'Inflexible');
        
        for (const rotator of inflexibleRotators) {
            for (const blockNumber of rotator.assignedBlocks) {
                const assignment = this.createRotationAssignment(
                    rotator.residentId,
                    blockNumber,
                    'External_Rotation_In',
                    0,
                    false, // not mandatory
                    false, // not exam leave
                    false  // not holiday leave
                );
                
                this.addAssignment(rotator.residentId, assignment);
                this.updateBlockCoverage(blockNumber, 'add', rotator);
                this.metrics.externalRotatorsPlaced++;
                
                console.log(`  ✅ Placed ${rotator.residentName} (${rotator.service}) in Block ${blockNumber} [Inflexible]`);
            }
        }
        
        // Then, strategically place flexible external rotators
        const flexibleRotators = this.externalRotators.filter(r => r.type === 'Flexible');
        
        for (const rotator of flexibleRotators) {
            const optimalBlocks = this.findOptimalBlocksForFlexibleRotator(rotator);
            
            for (const blockNumber of optimalBlocks) {
                const assignment = this.createRotationAssignment(
                    rotator.residentId,
                    blockNumber,
                    'External_Rotation_In',
                    0,
                    false,
                    false,
                    false
                );
                
                this.addAssignment(rotator.residentId, assignment);
                this.updateBlockCoverage(blockNumber, 'add', rotator);
                this.metrics.externalRotatorsPlaced++;
                this.metrics.flexibleRotatorsOptimized++;
                
                console.log(`  ✅ Optimally placed ${rotator.residentName} (${rotator.service}) in Block ${blockNumber} [Flexible]`);
            }
        }
        
        this.metrics.phaseDurations.set(0, Date.now() - phaseStart);
        console.log(`  ⏱️ Phase 0 completed in ${Date.now() - phaseStart}ms`);
    }

    /**
     * Find optimal blocks for flexible external rotator
     */
    private findOptimalBlocksForFlexibleRotator(rotator: ExternalRotatorProfile): number[] {
        const blocksNeeded = Math.min(3, rotator.availableBlocks.length); // Typically 3 blocks per rotator
        const optimalBlocks: number[] = [];
        
        // Sort available blocks by projected coverage (ascending)
        const sortedBlocks = [...rotator.availableBlocks].sort((a, b) => {
            const coverageA = this.blockCoverage.get(a)?.projectedCoverage || 999;
            const coverageB = this.blockCoverage.get(b)?.projectedCoverage || 999;
            return coverageA - coverageB;
        });
        
        // Select blocks with lowest coverage
        for (let i = 0; i < blocksNeeded && i < sortedBlocks.length; i++) {
            optimalBlocks.push(sortedBlocks[i]);
            
            // Update projected coverage
            const coverage = this.blockCoverage.get(sortedBlocks[i]);
            if (coverage) {
                coverage.projectedCoverage++;
            }
        }
        
        return optimalBlocks;
    }

    /**
     * Phase 1: Place Inflexible Off-Service Rotations
     */
    private async executePhase1_InflexibleOffService(): Promise<void> {
        const phaseStart = Date.now();
        console.log(`\n📌 Phase 1: Placing Inflexible Off-Service Rotations`);
        
        // Process mandatory rotations (e.g., "PGY-1s must do Trauma in Block 3")
        for (const mandatory of this.mandatoryRotations) {
            const eligibleResidents = this.residents.filter(r => 
                r.pgyLevel === mandatory.pgyLevel &&
                r.service === 'Neurosurgery' &&
                !this.hasAssignmentInBlock(r.id, mandatory.blockNumber)
            );
            
            for (const resident of eligibleResidents.slice(0, mandatory.requiredResidents)) {
                // Check for consecutive blocks requirement
                if (mandatory.consecutiveBlocks && mandatory.consecutiveBlocks > 1) {
                    const consecutiveAssignments = this.createConsecutiveBlockAssignments(
                        resident.id,
                        mandatory.blockNumber,
                        mandatory.consecutiveBlocks,
                        mandatory.rotationType,
                        1,
                        true // is mandatory
                    );
                    
                    consecutiveAssignments.forEach(assignment => {
                        this.addAssignment(resident.id, assignment);
                        this.updateBlockCoverage(assignment.blockNumber, 'remove', resident);
                    });
                    
                    this.metrics.consecutiveBlocksCreated++;
                    console.log(`  ✅ Assigned ${resident.name} to ${mandatory.rotationType} for ${mandatory.consecutiveBlocks} consecutive blocks starting Block ${mandatory.blockNumber}`);
                } else {
                    const assignment = this.createRotationAssignment(
                        resident.id,
                        mandatory.blockNumber,
                        mandatory.rotationType,
                        1,
                        true, // is mandatory
                        false,
                        false
                    );
                    
                    this.addAssignment(resident.id, assignment);
                    this.updateBlockCoverage(mandatory.blockNumber, 'remove', resident);
                    
                    console.log(`  ✅ Assigned ${resident.name} to ${mandatory.rotationType} in Block ${mandatory.blockNumber}`);
                }
            }
        }
        
        this.metrics.phaseDurations.set(1, Date.now() - phaseStart);
        console.log(`  ⏱️ Phase 1 completed in ${Date.now() - phaseStart}ms`);
    }

    /**
     * Phase 2: Place Mandatory Exam Leave
     */
    private async executePhase2_MandatoryExamLeave(): Promise<void> {
        const phaseStart = Date.now();
        console.log(`\n📚 Phase 2: Placing Mandatory Exam Leave`);
        
        for (const examRequirement of this.examLeaveRequirements) {
            const eligibleResidents = this.residents.filter(r => 
                r.pgyLevel === examRequirement.pgyLevel &&
                r.service === 'Neurosurgery'
            );
            
            for (const resident of eligibleResidents) {
                if (!this.hasAssignmentInBlock(resident.id, examRequirement.blockNumber)) {
                    const assignment = this.createRotationAssignment(
                        resident.id,
                        examRequirement.blockNumber,
                        'Exam_Leave',
                        2,
                        true, // is mandatory
                        true, // is exam leave
                        false
                    );
                    
                    this.addAssignment(resident.id, assignment);
                    this.updateBlockCoverage(examRequirement.blockNumber, 'remove', resident);
                    
                    console.log(`  ✅ Assigned ${resident.name} (PGY-${resident.pgyLevel}) to Exam Leave in Block ${examRequirement.blockNumber}`);
                }
            }
        }
        
        this.metrics.phaseDurations.set(2, Date.now() - phaseStart);
        console.log(`  ⏱️ Phase 2 completed in ${Date.now() - phaseStart}ms`);
    }

    /**
     * Phase 3: Assign Holiday Leave via Points System
     */
    private async executePhase3_HolidayLeave(): Promise<void> {
        const phaseStart = Date.now();
        console.log(`\n🎄 Phase 3: Assigning Holiday Leave via Points System`);
        
        // Find holiday blocks
        const christmasBlock = this.rotationBlocks.find(b => {
            const blockStart = b.startDate.toDate();
            const blockEnd = b.endDate.toDate();
            const christmas = new Date(blockStart.getFullYear(), 11, 25);
            return blockStart <= christmas && christmas <= blockEnd;
        });
        
        const newYearBlock = this.rotationBlocks.find(b => {
            const blockStart = b.startDate.toDate();
            const blockEnd = b.endDate.toDate();
            const newYear = new Date(blockEnd.getFullYear(), 0, 1);
            return blockStart <= newYear && newYear <= blockEnd;
        });
        
        // Sort residents by points for Christmas
        if (christmasBlock) {
            const christmasEligible = Array.from(this.holidayPoints.values())
                .filter(hp => hp.christmasEligible && !hp.awarded)
                .sort((a, b) => b.totalPoints - a.totalPoints);
            
            // Award Christmas to top scorers
            const christmasWinners = christmasEligible.slice(0, this.coverageRequirements.maxResidentsOnLeave);
            
            for (const winner of christmasWinners) {
                if (!this.hasAssignmentInBlock(winner.residentId, christmasBlock.blockNumber)) {
                    const assignment = this.createRotationAssignment(
                        winner.residentId,
                        christmasBlock.blockNumber,
                        'Holiday_Leave',
                        3,
                        false,
                        false,
                        true // is holiday leave
                    );
                    
                    this.addAssignment(winner.residentId, assignment);
                    this.updateBlockCoverage(christmasBlock.blockNumber, 'remove', 
                        this.residents.find(r => r.id === winner.residentId)!);
                    
                    winner.awarded = 'Christmas';
                    this.metrics.holidayAssignments++;
                    
                    console.log(`  🎁 Awarded Christmas leave to ${winner.residentName} (${winner.totalPoints} points)`);
                }
            }
        }
        
        // Sort residents by points for New Year
        if (newYearBlock) {
            const newYearEligible = Array.from(this.holidayPoints.values())
                .filter(hp => hp.newYearEligible && !hp.awarded)
                .sort((a, b) => b.totalPoints - a.totalPoints);
            
            // Award New Year to top scorers
            const newYearWinners = newYearEligible.slice(0, this.coverageRequirements.maxResidentsOnLeave);
            
            for (const winner of newYearWinners) {
                if (!this.hasAssignmentInBlock(winner.residentId, newYearBlock.blockNumber)) {
                    const assignment = this.createRotationAssignment(
                        winner.residentId,
                        newYearBlock.blockNumber,
                        'Holiday_Leave',
                        3,
                        false,
                        false,
                        true
                    );
                    
                    this.addAssignment(winner.residentId, assignment);
                    this.updateBlockCoverage(newYearBlock.blockNumber, 'remove',
                        this.residents.find(r => r.id === winner.residentId)!);
                    
                    winner.awarded = 'NewYear';
                    this.metrics.holidayAssignments++;
                    
                    console.log(`  🎊 Awarded New Year leave to ${winner.residentName} (${winner.totalPoints} points)`);
                }
            }
        }
        
        this.metrics.phaseDurations.set(3, Date.now() - phaseStart);
        console.log(`  ⏱️ Phase 3 completed in ${Date.now() - phaseStart}ms`);
    }

    /**
     * Phase 4: Fill Core Neurosurgery Rotations
     */
    private async executePhase4_CoreRotations(): Promise<void> {
        const phaseStart = Date.now();
        console.log(`\n🏥 Phase 4: Filling Core Neurosurgery Rotations`);
        
        for (let blockNumber = 1; blockNumber <= this.TOTAL_BLOCKS; blockNumber++) {
            const coverage = this.blockCoverage.get(blockNumber)!;
            
            // Get residents without assignments in this block
            const availableResidents = this.residents.filter(r => 
                r.service === 'Neurosurgery' &&
                !this.hasAssignmentInBlock(r.id, blockNumber)
            );
            
            // Calculate how many more neurosurgery residents we need
            const neededResidents = Math.max(
                0,
                this.coverageRequirements.minNeurosurgeryResidents - coverage.neurosurgeryResidents
            );
            
            // Prioritize by seniority and workload balance
            const prioritizedResidents = this.prioritizeResidentsForCore(availableResidents, blockNumber);
            
            // Assign core rotations
            for (let i = 0; i < Math.min(neededResidents, prioritizedResidents.length); i++) {
                const resident = prioritizedResidents[i];
                
                const assignment = this.createRotationAssignment(
                    resident.id,
                    blockNumber,
                    'Core_Neurosurgery',
                    4,
                    false,
                    false,
                    false
                );
                
                this.addAssignment(resident.id, assignment);
                this.updateBlockCoverage(blockNumber, 'add', resident);
                
                console.log(`  ✅ Assigned ${resident.name} to Core Neurosurgery in Block ${blockNumber}`);
            }
        }
        
        this.metrics.phaseDurations.set(4, Date.now() - phaseStart);
        console.log(`  ⏱️ Phase 4 completed in ${Date.now() - phaseStart}ms`);
    }

    /**
     * Phase 5: Place Flexible Rotations and Electives
     */
    private async executePhase5_FlexibleRotations(): Promise<void> {
        const phaseStart = Date.now();
        console.log(`\n🔄 Phase 5: Placing Flexible Rotations & Electives`);
        
        // Get all residents with unassigned blocks
        const residentsWithGaps = this.residents.filter(r => {
            const assignments = this.assignments.get(r.id) || [];
            return assignments.length < this.TOTAL_BLOCKS;
        });
        
        for (const resident of residentsWithGaps) {
            const assignments = this.assignments.get(resident.id) || [];
            const assignedBlocks = new Set(assignments.map(a => a.blockNumber));
            
            // Find unassigned blocks
            for (let blockNumber = 1; blockNumber <= this.TOTAL_BLOCKS; blockNumber++) {
                if (!assignedBlocks.has(blockNumber)) {
                    // Determine appropriate elective based on PGY level and interests
                    const electiveType = this.selectElectiveForResident(resident, blockNumber);
                    
                    const assignment = this.createRotationAssignment(
                        resident.id,
                        blockNumber,
                        electiveType,
                        5,
                        false,
                        false,
                        false
                    );
                    
                    this.addAssignment(resident.id, assignment);
                    
                    // Update coverage if it's a neurosurgery-related elective
                    if (electiveType === 'Research' || electiveType === 'Admin') {
                        this.updateBlockCoverage(blockNumber, 'partial', resident);
                    }
                    
                    console.log(`  ✅ Assigned ${resident.name} to ${electiveType} in Block ${blockNumber}`);
                }
            }
        }
        
        this.metrics.phaseDurations.set(5, Date.now() - phaseStart);
        console.log(`  ⏱️ Phase 5 completed in ${Date.now() - phaseStart}ms`);
    }

    /**
     * Phase 6: Balance Teams (Red vs Blue)
     */
    private async executePhase6_TeamBalancing(): Promise<void> {
        const phaseStart = Date.now();
        console.log(`\n⚖️ Phase 6: Balancing Teams (Red vs Blue)`);
        
        for (let blockNumber = 1; blockNumber <= this.TOTAL_BLOCKS; blockNumber++) {
            // Get all core neurosurgery assignments for this block
            const blockAssignments = this.getAllAssignmentsForBlock(blockNumber)
                .filter(a => a.rotationType === 'Core_Neurosurgery');
            
            if (blockAssignments.length === 0) continue;
            
            // Sort by seniority for balanced distribution
            const sortedAssignments = blockAssignments.sort((a, b) => {
                const residentA = this.residents.find(r => r.id === a.residentId);
                const residentB = this.residents.find(r => r.id === b.residentId);
                return (residentB?.pgyLevel || 0) - (residentA?.pgyLevel || 0);
            });
            
            // Alternate team assignments to ensure balance
            let redCount = 0;
            let blueCount = 0;
            
            sortedAssignments.forEach((assignment, index) => {
                // Distribute seniors evenly
                if (index % 2 === 0) {
                    assignment.teamAssignment = 'Red';
                    redCount++;
                } else {
                    assignment.teamAssignment = 'Blue';
                    blueCount++;
                }
            });
            
            // Update coverage tracking
            const coverage = this.blockCoverage.get(blockNumber);
            if (coverage) {
                coverage.redTeamCount = redCount;
                coverage.blueTeamCount = blueCount;
            }
            
            console.log(`  ✅ Block ${blockNumber}: Red Team (${redCount}), Blue Team (${blueCount})`);
        }
        
        // Calculate team balance score
        this.calculateTeamBalanceScore();
        
        this.metrics.phaseDurations.set(6, Date.now() - phaseStart);
        console.log(`  ⏱️ Phase 6 completed in ${Date.now() - phaseStart}ms`);
        console.log(`  📊 Team Balance Score: ${this.metrics.teamBalanceScore.toFixed(2)}/100`);
    }

    /**
     * Create a rotation assignment
     */
    private createRotationAssignment(
        residentId: string,
        blockNumber: number,
        rotationType: RotationType,
        phase: number,
        isMandatory: boolean,
        isExamLeave: boolean,
        isHolidayLeave: boolean
    ): YearlyRotationAssignment {
        const block = this.rotationBlocks[blockNumber - 1];
        
        return {
            id: `rotation-${uuidv4()}`,
            residentId,
            blockNumber,
            blockStartDate: block.startDate.toDate(),
            blockEndDate: block.endDate.toDate(),
            rotationType,
            isMandatory,
            isExamLeave,
            isHolidayLeave,
            isConsecutive: false,
            consecutiveBlockIds: [],
            phase,
            status: 'Scheduled',
            createdAt: admin.firestore.Timestamp.now(),
            updatedAt: admin.firestore.Timestamp.now()
        };
    }

    /**
     * Create consecutive block assignments
     */
    private createConsecutiveBlockAssignments(
        residentId: string,
        startBlock: number,
        numBlocks: number,
        rotationType: RotationType,
        phase: number,
        isMandatory: boolean
    ): YearlyRotationAssignment[] {
        const assignments: YearlyRotationAssignment[] = [];
        const consecutiveIds: string[] = [];
        
        for (let i = 0; i < numBlocks; i++) {
            const blockNumber = startBlock + i;
            if (blockNumber > this.TOTAL_BLOCKS) break;
            
            const assignment = this.createRotationAssignment(
                residentId,
                blockNumber,
                rotationType,
                phase,
                isMandatory,
                false,
                false
            );
            
            assignment.isConsecutive = true;
            consecutiveIds.push(assignment.id);
            assignments.push(assignment);
        }
        
        // Link consecutive blocks
        assignments.forEach(a => {
            a.consecutiveBlockIds = consecutiveIds;
        });
        
        return assignments;
    }

    /**
     * Add assignment to resident's schedule
     */
    private addAssignment(residentId: string, assignment: YearlyRotationAssignment): void {
        const residentAssignments = this.assignments.get(residentId) || [];
        residentAssignments.push(assignment);
        this.assignments.set(residentId, residentAssignments);
    }

    /**
     * Update block coverage tracking
     */
    private updateBlockCoverage(
        blockNumber: number,
        action: 'add' | 'remove' | 'partial',
        resident: Resident | ExternalRotatorProfile
    ): void {
        const coverage = this.blockCoverage.get(blockNumber);
        if (!coverage) return;
        
        if (action === 'add') {
            coverage.totalResidents++;
            
            if ('service' in resident) {
                if (resident.service === 'Neurosurgery') {
                    coverage.neurosurgeryResidents++;
                } else {
                    coverage.externalRotators++;
                }
            }
            
            if (resident.pgyLevel >= 4) {
                coverage.seniorResidents++;
            }
        } else if (action === 'remove') {
            coverage.projectedCoverage--;
            
            if ('service' in resident && resident.service === 'Neurosurgery') {
                coverage.neurosurgeryResidents--;
            }
        } else if (action === 'partial') {
            // For research/admin, they partially count toward coverage
            coverage.neurosurgeryResidents += 0.5;
        }
        
        // Check if requirements are met
        this.validateBlockCoverage(blockNumber);
    }

    /**
     * Validate block coverage meets requirements
     */
    private validateBlockCoverage(blockNumber: number): void {
        const coverage = this.blockCoverage.get(blockNumber);
        if (!coverage) return;
        
        coverage.deficits = [];
        coverage.meetsRequirements = true;
        
        if (coverage.totalResidents < this.coverageRequirements.minTotalResidents) {
            coverage.deficits.push(`Insufficient total residents: ${coverage.totalResidents}/${this.coverageRequirements.minTotalResidents}`);
            coverage.meetsRequirements = false;
        }
        
        if (coverage.neurosurgeryResidents < this.coverageRequirements.minNeurosurgeryResidents) {
            coverage.deficits.push(`Insufficient neurosurgery residents: ${coverage.neurosurgeryResidents}/${this.coverageRequirements.minNeurosurgeryResidents}`);
            coverage.meetsRequirements = false;
        }
        
        if (coverage.seniorResidents < this.coverageRequirements.minSeniorResidents) {
            coverage.deficits.push(`Insufficient senior residents: ${coverage.seniorResidents}/${this.coverageRequirements.minSeniorResidents}`);
            coverage.meetsRequirements = false;
        }
        
        if (!coverage.meetsRequirements) {
            this.metrics.coverageViolations++;
        }
    }

    /**
     * Check if resident has assignment in block
     */
    private hasAssignmentInBlock(residentId: string, blockNumber: number): boolean {
        const assignments = this.assignments.get(residentId) || [];
        return assignments.some(a => a.blockNumber === blockNumber);
    }

    /**
     * Get all assignments for a block
     */
    private getAllAssignmentsForBlock(blockNumber: number): YearlyRotationAssignment[] {
        const blockAssignments: YearlyRotationAssignment[] = [];
        
        this.assignments.forEach(residentAssignments => {
            residentAssignments
                .filter(a => a.blockNumber === blockNumber)
                .forEach(a => blockAssignments.push(a));
        });
        
        return blockAssignments;
    }

    /**
     * Prioritize residents for core rotation assignment
     */
    private prioritizeResidentsForCore(residents: Resident[], blockNumber: number): Resident[] {
        return residents.sort((a, b) => {
            // Priority 1: Senior residents (for supervision requirement)
            if (a.pgyLevel >= 4 && b.pgyLevel < 4) return -1;
            if (b.pgyLevel >= 4 && a.pgyLevel < 4) return 1;
            
            // Priority 2: Chiefs
            if (a.isChief && !b.isChief) return -1;
            if (b.isChief && !a.isChief) return 1;
            
            // Priority 3: Workload balance (fewer assignments so far)
            const aAssignments = this.assignments.get(a.id)?.length || 0;
            const bAssignments = this.assignments.get(b.id)?.length || 0;
            
            return aAssignments - bAssignments;
        });
    }

    /**
     * Select appropriate elective for resident
     */
    private selectElectiveForResident(resident: Resident, blockNumber: number): RotationType {
        // Logic to select elective based on PGY level and career goals
        if (resident.pgyLevel >= 4) {
            // Senior residents get research or admin time
            return blockNumber % 2 === 0 ? 'Research' : 'Admin';
        } else if (resident.pgyLevel === 3) {
            // PGY-3s get electives
            const electives: RotationType[] = ['Spine', 'Pediatric', 'Functional', 'Vascular'];
            return electives[blockNumber % electives.length];
        } else {
            // Junior residents get more core experience
            return 'Core_Neurosurgery';
        }
    }

    /**
     * Calculate team balance score
     */
    private calculateTeamBalanceScore(): void {
        let totalDeviation = 0;
        let blockCount = 0;
        
        this.blockCoverage.forEach(coverage => {
            if (coverage.redTeamCount > 0 || coverage.blueTeamCount > 0) {
                const deviation = Math.abs(coverage.redTeamCount - coverage.blueTeamCount);
                totalDeviation += deviation;
                blockCount++;
            }
        });
        
        // Score from 0-100 (100 = perfect balance)
        if (blockCount > 0) {
            const avgDeviation = totalDeviation / blockCount;
            this.metrics.teamBalanceScore = Math.max(0, 100 - (avgDeviation * 20));
        } else {
            this.metrics.teamBalanceScore = 100;
        }
    }

    /**
     * Validate the complete schedule
     */
    private validateSchedule(): void {
        console.log(`\n🔍 Validating final schedule...`);
        
        // Check each resident has full year coverage
        this.residents.forEach(resident => {
            const assignments = this.assignments.get(resident.id) || [];
            const assignedBlocks = new Set(assignments.map(a => a.blockNumber));
            
            if (assignedBlocks.size < this.TOTAL_BLOCKS) {
                const missingBlocks: number[] = [];
                for (let i = 1; i <= this.TOTAL_BLOCKS; i++) {
                    if (!assignedBlocks.has(i)) {
                        missingBlocks.push(i);
                    }
                }
                this.violations.push(`${resident.name} missing assignments for blocks: ${missingBlocks.join(', ')}`);
            }
        });
        
        // Check coverage requirements for each block
        this.blockCoverage.forEach((coverage, blockNumber) => {
            if (!coverage.meetsRequirements) {
                this.violations.push(`Block ${blockNumber} coverage violations: ${coverage.deficits.join('; ')}`);
            }
        });
        
        // Check consecutive block integrity
        this.assignments.forEach((residentAssignments, residentId) => {
            const consecutiveGroups = new Map<string, YearlyRotationAssignment[]>();
            
            residentAssignments
                .filter(a => a.isConsecutive)
                .forEach(a => {
                    const groupId = a.consecutiveBlockIds[0];
                    if (!consecutiveGroups.has(groupId)) {
                        consecutiveGroups.set(groupId, []);
                    }
                    consecutiveGroups.get(groupId)!.push(a);
                });
            
            consecutiveGroups.forEach((group, groupId) => {
                const blocks = group.map(a => a.blockNumber).sort((a, b) => a - b);
                for (let i = 1; i < blocks.length; i++) {
                    if (blocks[i] !== blocks[i - 1] + 1) {
                        const resident = this.residents.find(r => r.id === residentId);
                        this.violations.push(`${resident?.name} has non-consecutive blocks in group ${groupId}`);
                    }
                }
            });
        });
    }

    /**
     * Get all assignments
     */
    private getAllAssignments(): YearlyRotationAssignment[] {
        const allAssignments: YearlyRotationAssignment[] = [];
        
        this.assignments.forEach(residentAssignments => {
            residentAssignments.forEach(assignment => {
                allAssignments.push(assignment);
            });
        });
        
        return allAssignments.sort((a, b) => {
            if (a.blockNumber !== b.blockNumber) {
                return a.blockNumber - b.blockNumber;
            }
            return a.residentId.localeCompare(b.residentId);
        });
    }

    /**
     * Get schedule analytics
     */
    public getAnalytics(): any {
        const analytics = {
            academicYear: this.academicYear.name,
            totalResidents: this.residents.length,
            totalBlocks: this.TOTAL_BLOCKS,
            metrics: {
                totalAssignments: this.metrics.totalAssignments,
                externalRotatorsPlaced: this.metrics.externalRotatorsPlaced,
                flexibleRotatorsOptimized: this.metrics.flexibleRotatorsOptimized,
                holidayAssignments: this.metrics.holidayAssignments,
                consecutiveBlocksCreated: this.metrics.consecutiveBlocksCreated,
                coverageViolations: this.metrics.coverageViolations,
                teamBalanceScore: this.metrics.teamBalanceScore,
                generationTimeMs: this.metrics.generationTimeMs
            },
            phaseTimings: Array.from(this.metrics.phaseDurations.entries()).map(([phase, duration]) => ({
                phase,
                name: this.PHASES[phase as keyof typeof this.PHASES],
                durationMs: duration
            })),
            coverageAnalysis: Array.from(this.blockCoverage.values()).map(coverage => ({
                block: coverage.blockNumber,
                totalResidents: coverage.totalResidents,
                neurosurgeryResidents: coverage.neurosurgeryResidents,
                seniorResidents: coverage.seniorResidents,
                externalRotators: coverage.externalRotators,
                redTeam: coverage.redTeamCount,
                blueTeam: coverage.blueTeamCount,
                meetsRequirements: coverage.meetsRequirements,
                deficits: coverage.deficits
            })),
            holidayAwards: Array.from(this.holidayPoints.values())
                .filter(hp => hp.awarded)
                .map(hp => ({
                    resident: hp.residentName,
                    points: hp.totalPoints,
                    awarded: hp.awarded
                })),
            violations: this.violations
        };
        
        return analytics;
    }

    /**
     * Export schedule to various formats
     */
    public exportSchedule(format: 'json' | 'csv' | 'grid' = 'json'): string {
        switch (format) {
            case 'csv':
                return this.exportToCSV();
            case 'grid':
                return this.exportToGrid();
            default:
                return this.exportToJSON();
        }
    }

    /**
     * Export to JSON format
     */
    private exportToJSON(): string {
        return JSON.stringify({
            metadata: {
                academicYear: this.academicYear.name,
                startDate: this.academicYear.startDate.toDate().toISOString(),
                endDate: this.academicYear.endDate.toDate().toISOString(),
                generated: new Date().toISOString(),
                generatedBy: 'ramihatou97',
                version: '1.0.0'
            },
            assignments: this.getAllAssignments(),
            analytics: this.getAnalytics()
        }, null, 2);
    }

    /**
     * Export to CSV format
     */
    private exportToCSV(): string {
        const headers = ['Resident', 'PGY', 'Block', 'Start Date', 'End Date', 'Rotation', 'Team', 'Type'];
        const rows = [headers.join(',')];
        
        this.getAllAssignments().forEach(assignment => {
            const resident = this.residents.find(r => r.id === assignment.residentId);
            
            rows.push([
                resident?.name || 'Unknown',
                resident?.pgyLevel || '',
                assignment.blockNumber,
                assignment.blockStartDate.toLocaleDateString(),
                assignment.blockEndDate.toLocaleDateString(),
                assignment.rotationType,
                assignment.teamAssignment || '',
                assignment.isMandatory ? 'Mandatory' : assignment.isHolidayLeave ? 'Holiday' : 'Regular'
            ].join(','));
        });
        
        return rows.join('\n');
    }

    /**
     * Export to grid format (visual representation)
     */
    private exportToGrid(): string {
        const grid: string[] = [];
        
        // Header row
        const headerRow = ['Resident/Block'];
        for (let i = 1; i <= this.TOTAL_BLOCKS; i++) {
            headerRow.push(`B${i}`);
        }
        grid.push(headerRow.join('\t'));
        
        // Resident rows
        this.residents
            .filter(r => r.service === 'Neurosurgery')
            .sort((a, b) => b.pgyLevel - a.pgyLevel)
            .forEach(resident => {
                const row = [`${resident.name} (PGY-${resident.pgyLevel})`];
                const assignments = this.assignments.get(resident.id) || [];
                
                for (let blockNum = 1; blockNum <= this.TOTAL_BLOCKS; blockNum++) {
                    const assignment = assignments.find(a => a.blockNumber === blockNum);
                    
                    if (assignment) {
                        let cellContent = this.getRotationAbbreviation(assignment.rotationType);
                        if (assignment.teamAssignment) {
                            cellContent += `-${assignment.teamAssignment[0]}`;
                        }
                        row.push(cellContent);
                    } else {
                        row.push('-');
                    }
                }
                
                grid.push(row.join('\t'));
            });
        
        return grid.join('\n');
    }

    /**
     * Get rotation abbreviation for grid display
     */
    private getRotationAbbreviation(rotationType: RotationType): string {
        const abbreviations: Record<string, string> = {
            'Core_Neurosurgery': 'CORE',
            'Trauma': 'TRMA',
            'ICU': 'ICU',
            'Research': 'RSCH',
            'Admin': 'ADMN',
            'Spine': 'SPNE',
            'Pediatric': 'PEDS',
            'Functional': 'FUNC',
            'Vascular': 'VASC',
            'Holiday_Leave': 'HDAY',
            'Exam_Leave': 'EXAM',
            'External_Rotation_In': 'EXT',
            'External_Rotation_Out': 'OUT'
        };
        
        return abbreviations[rotationType] || rotationType.substring(0, 4).toUpperCase();
    }
}

// Export the scheduler
export default YearlyRotationScheduler;