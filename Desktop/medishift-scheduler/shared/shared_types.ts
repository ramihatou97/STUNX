/**
 * MediShift Type Definitions
 * Version: 2.0 - Complete Implementation
 * Date: August 2025
 * 
 * Central type definitions for the entire application
 */

import { Timestamp } from 'firebase/firestore';

// ===================================================================
// 1. CORE USER & CONFIGURATION ENTITIES
// ===================================================================

export interface Resident {
    id: string;
    name: string;
    email: string;
    phone?: string;
    pgyLevel: 1 | 2 | 3 | 4 | 5 | 6 | 7;
    specialty: 'Neurosurgery' | 'Plastics' | 'Orthopedics' | 'General';
    onService: boolean;
    isChief: boolean;
    callExempt: boolean;
    team?: 'Red' | 'Blue';
    profilePhotoUrl?: string;
    startDate: Timestamp;
    graduationDate: Timestamp;
    preferences?: ResidentPreferences;
    stats?: ResidentStats;
}

export interface ResidentPreferences {
    preferredCallDays?: string[];
    avoidCallDays?: string[];
    maxCallsPerMonth?: number;
    notes?: string;
}

export interface ResidentStats {
    totalCallsThisYear: number;
    totalORHours: number;
    epasCompleted: number;
    averageCallsPerMonth: number;
    lastCallDate?: Timestamp;
}

export interface ExternalRotator {
    id: string;
    name: string;
    email: string;
    homeInstitution: string;
    homeService: string;
    pgyLevel: number;
    specialty: string;
    isEligibleForCall: boolean;
    startDate: Timestamp;
    endDate: Timestamp;
    assignedBlock: number;
    supervisorId?: string;
}

export interface Faculty {
    id: string;
    name: string;
    email: string;
    role: 'Attending' | 'Fellow' | 'Chief';
    specialty: string;
    canSuperviseEPAs: boolean;
    clinicDays?: string[];
    orDays?: string[];
}

// ===================================================================
// 2. CONFIGURATION & RULES
// ===================================================================

export interface AppConfiguration {
    // Coverage Rules
    coverageRules: {
        rotationBlock: CoverageRule[];
        weekday: WeekdayRule[];
        weekend: WeekendRule[];
        holiday: HolidayRule[];
    };
    
    // Monthly Scheduler Configuration
    monthlySchedulerConfig: {
        callRatios: { [pgyLevel: number]: number };
        paroHardCaps: PARORule[];
        maxWeekendsPerRotation: number;
        weekendDefinition: string[];
        minRestHours: number;
        maxConsecutiveCalls: number;
    };
    
    // Yearly Scheduler Configuration
    yearlySchedulerConfig: {
        mandatoryRotations: MandatoryRotationRule[];
        examLeave: ExamLeaveRule[];
        blockDuration: number;
        totalBlocksPerYear: number;
    };
    
    // Leave Policy
    leavePolicy: {
        minNoticeDays: number;
        maxConsecutiveDays: number;
        annualLimit: number;
        carryOverLimit: number;
        blackoutDates?: string[];
    };
    
    // System Settings
    systemSettings: {
        timezone: string;
        academicYearStart: string;
        notificationSettings: NotificationSettings;
    };
    
    // Holidays
    holidays: string[];
}

export interface CoverageRule {
    id: string;
    name: string;
    description: string;
    isEnabled: boolean;
    appliesTo: 'ALL' | 'SPECIALTY' | 'SPECIALTY_PGY_MIN';
    specialty?: string;
    minPgyLevel?: number;
    minCount: number;
    maxCount?: number;
    priority: number;
}

export interface WeekdayRule {
    dayOfWeek: number;
    minCoverage: number;
    optimalCoverage: number;
    maxCoverage: number;
}

export interface WeekendRule {
    minSeniors: number;
    minJuniors: number;
    requiresChief: boolean;
}

export interface HolidayRule {
    holidayName: string;
    date: string;
    minCoverage: number;
    bonusPoints: number;
}

export interface PARORule {
    minDays: number;
    maxDays: number;
    calls: number;
}

export interface MandatoryRotationRule {
    rotationName: string;
    pgyLevels: number[];
    blockNumber: number;
    duration: number;
}

export interface ExamLeaveRule {
    examName: string;
    pgyLevels: number[];
    blockNumber: number;
    duration: number;
}

export interface NotificationSettings {
    emailEnabled: boolean;
    pushEnabled: boolean;
    smsEnabled: boolean;
    reminderDays: number[];
}

// ===================================================================
// 3. YEARLY ROTATION SCHEDULER
// ===================================================================

export interface AcademicYear {
    id: string; // "2025-2026"
    blocks: RotationBlock[];
    metadata: {
        generatedAt: Timestamp;
        generatedBy?: string;
        totalResidents: number;
        totalExternalRotators: number;
        version: string;
        approvedAt?: Timestamp;
        approvedBy?: string;
    };
}

export interface RotationBlock {
    blockNumber: number;
    name?: string;
    startDate: Timestamp;
    endDate: Timestamp;
    assignments: RotationAssignment[];
    externalRotators?: string[];
    notes?: string;
}

export interface RotationAssignment {
    residentId: string;
    residentName?: string;
    rotationName: string;
    rotationType: 
        | 'CORE_NSX' 
        | 'MANDATORY_OFF_SERVICE' 
        | 'FLEXIBLE_OFF_SERVICE' 
        | 'RESEARCH' 
        | 'EXAM_LEAVE' 
        | 'HOLIDAY_LEAVE'
        | 'ELECTIVE';
    location?: string;
    supervisor?: string;
    team?: 'Red' | 'Blue';
    required?: boolean;
    holidayType?: 'Christmas' | 'NewYear' | 'Other';
}

// ===================================================================
// 4. VACATION & LEAVE SYSTEM
// ===================================================================

export interface LeaveRequest {
    id: string;
    residentId: string;
    residentName: string;
    type: 'Personal' | 'Professional' | 'LieuDay' | 'Compassionate' | 'Sick' | 'Parental';
    startDate: Timestamp;
    endDate: Timestamp;
    reason?: string;
    status: 
        | 'Draft'
        | 'Pending Analysis' 
        | 'Pending Approval' 
        | 'Approved' 
        | 'Denied' 
        | 'ApprovedWithConflict'
        | 'Cancelled'
        | 'Analysis Failed';
    
    // Analysis & Conflict
    analysisReportId?: string;
    conflictTicketId?: string;
    denialJustification?: string;
    
    // Metadata
    createdAt: Timestamp;
    updatedAt?: Timestamp;
    submittedAt?: Timestamp;
    reviewedAt?: Timestamp;
    reviewedBy?: string;
    
    // Coverage
    coverageArrangements?: CoverageArrangement[];
    attachments?: string[];
}

export interface CoverageArrangement {
    date: Timestamp;
    coveringResidentId: string;
    coveringResidentName: string;
    dutyType: 'Call' | 'OR' | 'Clinic' | 'All';
    confirmed: boolean;
}

export interface LeaveAnalysisReport {
    id: string;
    requestId: string;
    residentId: string;
    residentName: string;
    analyzedAt: Timestamp;
    
    // Recommendation
    overallRecommendation: 'Approve' | 'Flagged for Review' | 'Deny';
    denialReason?: string;
    
    // Coverage Analysis
    estimatedCoverageImpact: {
        projectedCoverageRisk: 'Low' | 'Medium' | 'High';
        availableResidents: number;
        coverageRatio: number;
        criticalDates: Timestamp[];
    };
    
    // Fairness Analysis
    fairnessScore: {
        score: number; // 0-100
        historicalSuccessRateForPeriod: number;
        recentDaysOff: number;
        peerComparison: number;
    };
    
    // Conflicts
    scheduleConflicts: ScheduleConflict[];
    
    // Alternatives
    alternativeDates?: Date[];
    notes?: string[];
}

export interface ScheduleConflict {
    type: 'Call' | 'OR' | 'Clinic' | 'Teaching' | 'Exam';
    date: Timestamp;
    description: string;
    severity: 'Low' | 'Medium' | 'High';
    resolvable: boolean;
}

export interface ConflictResolutionTicket {
    id: string;
    leaveRequestId: string;
    residentId: string;
    residentName: string;
    conflictStartDate: Timestamp;
    conflictEndDate: Timestamp;
    conflictingAssignments: ConflictingAssignment[];
    status: 'Open' | 'In Progress' | 'Resolved' | 'Escalated';
    priority?: 'Low' | 'Medium' | 'High';
    assignedTo?: string;
    resolution?: string;
    createdAt: Timestamp;
    resolvedAt?: Timestamp;
}

export interface ConflictingAssignment {
    type: 'Call' | 'OR' | 'Clinic';
    description: string;
    date: Timestamp;
    originalAssignee?: string;
    replacementAssignee?: string;
}

// ===================================================================
// 5. MONTHLY CALL SCHEDULER
// ===================================================================

export interface MonthlySchedule {
    id: string; // "2025-08"
    month: number;
    year: number;
    assignments: CallAssignment[];
    metadata: {
        generatedAt: Timestamp;
        generatedBy: string;
        staffingLevel: 'Normal' | 'Shortage';
        totalCalls: number;
        averageCallsPerResident: number;
        fairnessIndex: number;
    };
    published: boolean;
    publishedAt?: Timestamp;
}

export interface CallAssignment {
    id: string;
    residentId: string;
    residentName: string;
    date: Timestamp;
    type: 'Night' | 'Weekend' | 'Holiday' | 'PostCall' | 'Backup';
    points: number;
    isHoliday: boolean;
    team?: 'Red' | 'Blue';
    location?: string;
    
    // Status
    status?: 'Scheduled' | 'Confirmed' | 'Completed' | 'Swapped' | 'Cancelled';
    
    // Swaps
    swappedFrom?: string;
    swappedTo?: string;
    swapApprovedBy?: string;
    
    // Metadata
    createdAt: Timestamp;
    createdBy: string;
    modifiedAt?: Timestamp;
    modifiedBy?: string;
    notes?: string;
}

// ===================================================================
// 6. WEEKLY CLINICAL SCHEDULER
// ===================================================================

export interface WeeklySchedule {
    id: string; // "2025-W33"
    weekNumber: number;
    year: number;
    startDate: Timestamp;
    endDate: Timestamp;
    days: DailySchedule[];
    published: boolean;
    publishedAt?: Timestamp;
}

export interface DailySchedule {
    date: Timestamp;
    dayOfWeek: number;
    isHoliday: boolean;
    assignments: {
        or: ORAssignment[];
        clinic: ClinicAssignment[];
        call: CallAssignment[];
        float: FloatAssignment[];
        pager: PagerAssignment[];
    };
}

export interface ORAssignment {
    id: string;
    residentId: string;
    residentName: string;
    date: Timestamp;
    startTime: string;
    endTime: string;
    room: string;
    caseType?: string;
    surgeonId: string;
    surgeonName: string;
    procedureName?: string;
    isSpineCase?: boolean;
    duration?: number; // minutes
    status: 'Scheduled' | 'In Progress' | 'Completed' | 'Cancelled';
    notes?: string;
}

export interface ClinicAssignment {
    id: string;
    residentId: string;
    residentName: string;
    date: Timestamp;
    startTime: string;
    endTime: string;
    clinicType: string;
    clinicLocation: string;
    attendingId: string;
    attendingName: string;
    patientCount?: number;
    status: 'Scheduled' | 'Completed' | 'Cancelled';
}

export interface FloatAssignment {
    id: string;
    residentId: string;
    residentName: string;
    date: Timestamp;
    priority: number;
    assignedTo?: 'OR' | 'Clinic' | 'Consults';
}

export interface PagerAssignment {
    id: string;
    residentId: string;
    residentName: string;
    date: Timestamp;
    pagerType: 'Primary' | 'Backup';
    startTime: string;
    endTime: string;
}

// ===================================================================
// 7. OR CASES & SCHEDULING
// ===================================================================

export interface ORCase {
    id: string;
    caseId: string;
    date: Timestamp;
    scheduledStart: string;
    scheduledEnd: string;
    actualStart?: string;
    actualEnd?: string;
    
    // Surgical Team
    primarySurgeonId: string;
    primarySurgeonName: string;
    assistingSurgeons?: string[];
    assignedResidentIds: string[];
    assignedResidentNames?: string[];
    
    // Case Details
    procedureName: string;
    procedureType: string;
    procedureKeywords: string[];
    complexity: 'Low' | 'Medium' | 'High';
    isSpineCase: boolean;
    isEmergency: boolean;
    
    // Patient
    patientMRN?: string;
    patientAge?: number;
    
    // Room & Equipment
    orRoom: string;
    equipment?: string[];
    
    // Status
    status: 'Scheduled' | 'In Progress' | 'Completed' | 'Cancelled' | 'Postponed';
    
    // EPA Tracking
    epasGenerated?: boolean;
    associatedEPAs?: string[];
    
    // Notes
    preOpNotes?: string;
    postOpNotes?: string;
    complications?: string;
}

export interface ORSlot {
    id: string;
    date: Timestamp;
    room: string;
    startTime: string;
    endTime: string;
    surgeonId: string;
    surgeonName: string;
    isAvailable: boolean;
    priority: number;
    requiredResidentCount: number;
}

export interface ClinicSlot {
    id: string;
    date: Timestamp;
    clinicName: string;
    location: string;
    startTime: string;
    endTime: string;
    attendingId: string;
    attendingName: string;
    expectedVolume: number;
    requiredResidentCount: number;
}

// ===================================================================
// 8. EPA (ENTRUSTABLE PROFESSIONAL ACTIVITIES)
// ===================================================================

export interface EPADefinition {
    id: string;
    name: string;
    description: string;
    category: string;
    procedureKeywords: string[];
    minObservations: number;
    competencyLevel: 1 | 2 | 3 | 4 | 5;
    applicablePGYLevels: number[];
    milestones: string[];
    assessmentCriteria: AssessmentCriterion[];
}

export interface AssessmentCriterion {
    id: string;
    description: string;
    weight: number;
    required: boolean;
}

export interface EPAssignment {
    id: string;
    epaId: string;
    epaName: string;
    residentId: string;
    residentName?: string;
    pgyLevel?: number;
    
    // Case Association
    caseId: string;
    procedureName?: string;
    caseDate?: Timestamp;
    
    // Assessment
    assessorId: string;
    assessorName?: string;
    status: 'Assigned' | 'In Progress' | 'Completed' | 'Expired';
    
    // Scoring
    competencyLevel?: 1 | 2 | 3 | 4 | 5;
    scores?: { [criterionId: string]: number };
    overallScore?: number;
    
    // Feedback
    feedback?: string;
    strengths?: string[];
    improvements?: string[];
    
    // Dates
    assignedAt: Timestamp;
    dueDate: Timestamp;
    completedAt?: Timestamp;
    
    // Metadata
    attempts?: number;
    previousScores?: number[];
}

// ===================================================================
// 9. ANALYTICS & REPORTING
// ===================================================================

export interface AnalyticsReport {
    id: string;
    generatedDate: Timestamp;
    period: 'weekly' | 'monthly' | 'quarterly' | 'annual';
    startDate: Timestamp;
    endDate: Timestamp;
    
    // Overall Metrics
    overallMetrics: {
        totalResidents: number;
        totalCallsScheduled: number;
        totalORHours: number;
        totalClinicHours: number;
        callFairnessGini: number;
        averageCallsPerResident: number;
        
        // Leave Metrics
        totalLeaveRequests: number;
        approvedLeaveRequests: number;
        deniedLeaveRequests: number;
        leaveDenialRate: number;
        averageLeaveDays: number;
        totalLieuDaysOwed: number;
        
        // EPA Metrics
        totalEPAsAssigned: number;
        totalEPAsCompleted: number;
        averageCompletionRate: number;
        averageCompetencyScore: number;
        
        // Coverage Metrics
        coverageRate: number;
        understaffedDays: number;
        overstaffedDays: number;
    };
    
    // Individual Resident Metrics
    residentMetrics: ResidentMetric[];
    
    // Predictive Analytics
    predictiveMetrics: {
        projectedMonthlyORHours: number;
        projectedCallDistribution: number[];
        residentsAtRiskEpaCompletion: RiskAssessment[];
        expectedLeaveRequests: number;
        coverageForecast: CoverageForecast[];
    };
    
    // Trends
    trends?: {
        callDistributionTrend: number[];
        orHoursTrend: number[];
        epaCompletionTrend: number[];
        leaveRequestTrend: number[];
    };
    
    // Recommendations
    recommendations?: string[];
    warnings?: string[];
}

export interface ResidentMetric {
    residentId: string;
    residentName: string;
    pgyLevel: number;
    
    // Call Metrics
    totalCalls: number;
    nightCalls: number;
    weekendCalls: number;
    holidayCalls: number;
    callPoints: number;
    
    // Clinical Metrics
    totalORHours: number;
    spineCaseHours: number;
    totalClinicHours: number;
    uniqueProcedures: number;
    
    // EPA Metrics
    epasAssigned: number;
    epasCompleted: number;
    averageEPAScore: number;
    
    // Leave Metrics
    daysOffTaken: number;
    lieuDaysEarned: number;
    lieuDaysUsed: number;
    
    // Compliance
    paroCompliant: boolean;
    educationGoalsMet: boolean;
}

export interface RiskAssessment {
    residentId: string;
    residentName: string;
    riskType: 'EPA' | 'Clinical' | 'Wellness' | 'Academic';
    riskLevel: 'Low' | 'Medium' | 'High';
    projectedCompletion: number;
    recommendedAction: string;
}

export interface CoverageForecast {
    date: Timestamp;
    expectedCoverage: number;
    minimumRequired: number;
    surplus: number;
    risk: 'None' | 'Low' | 'Medium' | 'High';
}

// ===================================================================
// 10. NOTIFICATIONS
// ===================================================================

export interface Notification {
    id: string;
    recipientId: string;
    recipientEmail?: string;
    title: string;
    message: string;
    type: 'LeaveRequest' | 'EPA' | 'Conflict' | 'Schedule' | 'System' | 'Reminder';
    priority?: 'Low' | 'Medium' | 'High';
    
    // Navigation
    linkTo?: string;
    actionRequired?: boolean;
    
    // Status
    isRead: boolean;
    isArchived?: boolean;
    
    // Metadata
    createdAt: Timestamp;
    readAt?: Timestamp;
    expiresAt?: Timestamp;
    
    // Additional Data
    metadata?: Record<string, any>;
}

// ===================================================================
// 11. SWAP REQUESTS
// ===================================================================

export interface SwapRequest {
    id: string;
    requesterId: string;
    requesterName: string;
    targetResidentId: string;
    targetResidentName: string;
    
    // What's being swapped
    originalAssignment: {
        type: 'Call' | 'OR' | 'Clinic';
        date: Timestamp;
        details: string;
    };
    
    proposedAssignment: {
        type: 'Call' | 'OR' | 'Clinic';
        date: Timestamp;
        details: string;
    };
    
    // Status
    status: 'Pending' | 'Accepted' | 'Rejected' | 'Approved' | 'Cancelled';
    
    // Approval
    targetResidentResponse?: 'Accept' | 'Reject';
    targetRespondedAt?: Timestamp;
    adminApproval?: 'Approved' | 'Denied';
    adminApprovedBy?: string;
    adminApprovedAt?: Timestamp;
    
    // Metadata
    reason?: string;
    createdAt: Timestamp;
    expiresAt: Timestamp;
    notes?: string;
}

// ===================================================================
// 12. USER PROFILES & AUTH
// ===================================================================

export interface UserProfile {
    uid: string;
    email: string;
    displayName: string;
    photoURL?: string;
    role: 'Resident' | 'Admin' | 'Faculty' | 'ChiefResident' | 'ProgramDirector';
    residentId?: string; // Links to Resident entity
    facultyId?: string; // Links to Faculty entity
    
    // Preferences
    preferences: {
        emailNotifications: boolean;
        pushNotifications: boolean;
        smsNotifications: boolean;
        theme: 'light' | 'dark' | 'auto';
        language: string;
        timezone: string;
    };
    
    // Security
    lastLogin: Timestamp;
    createdAt: Timestamp;
    updatedAt: Timestamp;
    isActive: boolean;
    isEmailVerified: boolean;
    twoFactorEnabled?: boolean;
}

// ===================================================================
// 13. AUDIT & COMPLIANCE
// ===================================================================

export interface AuditLog {
    id: string;
    timestamp: Timestamp;
    userId: string;
    userName: string;
    action: string;
    entityType: string;
    entityId: string;
    changes?: Record<string, any>;
    ipAddress?: string;
    userAgent?: string;
    result: 'Success' | 'Failure';
    errorMessage?: string;
}

export interface ComplianceReport {
    id: string;
    period: string;
    generatedAt: Timestamp;
    
    // PARO Compliance
    paroViolations: PAROViolation[];
    workHourViolations: WorkHourViolation[];
    
    // Education Compliance
    epaCompletionRate: number;
    clinicalExposureMetrics: Record<string, number>;
    
    // Overall
    overallCompliance: boolean;
    issues: string[];
    recommendations: string[];
}

export interface PAROViolation {
    residentId: string;
    residentName: string;
    violationType: string;
    date: Timestamp;
    details: string;
    severity: 'Minor' | 'Major' | 'Critical';
}

export interface WorkHourViolation {
    residentId: string;
    residentName: string;
    week: string;
    hoursWorked: number;
    maxAllowed: number;
    consecutiveDays: number;
}

// ===================================================================
// TYPE GUARDS (Utility Functions)
// ===================================================================

export const isResident = (user: any): user is Resident => {
    return user && typeof user.pgyLevel === 'number';
};

export const isApprovedLeave = (leave: LeaveRequest): boolean => {
    return leave.status === 'Approved';
};

export const isOnCall = (assignment: any): assignment is CallAssignment => {
    return assignment && assignment.type && ['Night', 'Weekend', 'Holiday'].includes(assignment.type);
};

export const isCompleteEPA = (epa: EPAssignment): boolean => {
    return epa.status === 'Completed' && epa.overallScore !== undefined;
};