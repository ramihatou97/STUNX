import * as admin from 'firebase-admin';
import { MonthlyCallScheduler } from './scheduling/monthly-scheduler';
import { 
    Resident, 
    CallAssignment, 
    AppConfiguration, 
    AcademicYear,
    LeaveRequest,
    CrossMonthPostCall
} from '../../shared/types';

/**
 * Complete service implementation for scheduling functionality with robust batch processing
 */
export class SchedulerService {
    private db: FirebaseFirestore.Firestore;
    private readonly MAX_BATCH_SIZE = 400;
    private readonly MAX_RETRY_ATTEMPTS = 3;
    
    constructor() {
        this.db = admin.firestore();
    }
    
    /**
     * Generate a monthly schedule with comprehensive validation, error handling and batch processing
     */
    public async generateMonthlySchedule(
        month: number,
        year: number,
        staffingLevel: 'Normal' | 'Shortage' = 'Normal',
        userId: string,
        forceRegenerate: boolean = false
    ): Promise<{
        success: boolean;
        scheduleId?: string;
        assignments?: number;
        summary?: any;
        message?: string;
        cached?: boolean;
    }> {
        console.log(`📅 Generating schedule for ${month + 1}/${year} by ${userId}`);
        console.log(`🕒 Current time: ${new Date().toISOString()}`);
        
        try {
            // Validate inputs
            this.validateInputs(month, year);
            
            // Validate user ID to prevent database path issues
            this.validateUserId(userId);
            
            // Check if schedule already exists
            if (!forceRegenerate) {
                const existingSchedule = await this.db
                    .collection('monthlySchedules')
                    .where('month', '==', month)
                    .where('year', '==', year)
                    .limit(1)
                    .get();
                
                if (!existingSchedule.empty) {
                    console.log('Schedule already exists, returning existing');
                    return {
                        success: true,
                        scheduleId: existingSchedule.docs[0].id,
                        message: 'Schedule already exists',
                        cached: true
                    };
                }
            }
            
            // Get required data for scheduler
            const [
                residentsSnapshot, 
                configSnapshot, 
                academicYearSnapshot, 
                leaveSnapshot,
                existingAssignmentsSnapshot,
                previousMonthPostCallsSnapshot
            ] = await Promise.all([
                this.db.collection('residents').where('status', '==', 'Active').get(),
                this.db.collection('configuration').doc('app').get(),
                this.db.collection('academicYears').where('isCurrent', '==', true).limit(1).get(),
                this.db.collection('leaveRequests')
                    .where('status', '==', 'Approved')
                    .where('startDate', '<=', admin.firestore.Timestamp.fromDate(
                        new Date(year, month + 1, 0)
                    ))
                    .where('endDate', '>=', admin.firestore.Timestamp.fromDate(
                        new Date(year, month, 1)
                    ))
                    .get(),
                this.getPreviousAssignments(year, month),
                this.getPreviousMonthPostCalls(year, month)
            ]);
            
            // Validate required data
            this.validateRequiredData(residentsSnapshot, configSnapshot, academicYearSnapshot);
            
            // Prepare data for scheduler
            const residents = residentsSnapshot.docs.map(doc => ({
                id: doc.id,
                ...doc.data()
            })) as Resident[];
            
            const config = configSnapshot.data() as AppConfiguration;
            const academicYear = academicYearSnapshot.docs[0].data() as AcademicYear;
            const approvedLeave = leaveSnapshot.docs.map(doc => ({
                id: doc.id,
                ...doc.data()
            })) as LeaveRequest[];
            
            const existingAssignments = existingAssignmentsSnapshot.docs.map(doc => ({
                id: doc.id,
                ...doc.data()
            })) as CallAssignment[];
            
            const previousMonthPostCalls = previousMonthPostCallsSnapshot.docs.map(doc => ({
                id: doc.id,
                ...doc.data()
            })) as CrossMonthPostCall[];
            
            console.log(`📊 Data loaded: ${residents.length} residents, ${approvedLeave.length} leave requests`);
            
            // Create scheduler instance
            const scheduler = new MonthlyCallScheduler(
                residents,
                config,
                academicYear,
                approvedLeave,
                month,
                year,
                existingAssignments,
                previousMonthPostCalls
            );
            
            // Generate schedule
            const assignments = await scheduler.generateSchedule(staffingLevel);
            
            // Get cross-month post-calls
            const crossMonthPostCalls = scheduler.getCrossMonthPostCalls();
            
            // Get schedule summary
            const summary = scheduler.getScheduleSummary();
            
            // Save using batched writes to avoid transaction limits
            const scheduleId = await this.saveScheduleWithBatches(
                month,
                year,
                staffingLevel,
                userId,
                assignments,
                crossMonthPostCalls,
                summary,
                scheduler.getPerformanceMetrics()
            );
            
            // Create audit log
            await this.db.collection('auditLogs').add({
                action: 'generate_monthly_schedule',
                userId: userId,
                userIdSafe: this.getSafeUserId(userId),
                timestamp: admin.firestore.FieldValue.serverTimestamp(),
                details: {
                    month,
                    year,
                    staffingLevel,
                    assignmentCount: assignments.length,
                    scheduleId,
                    crossMonthPostCalls: crossMonthPostCalls.length,
                    emergencyAssignments: summary.emergencyAssignments || 0
                }
            });
            
            // Release memory
            scheduler.releaseMemory();
            
            console.log(`✅ Schedule generated successfully with ${assignments.length} assignments`);
            
            return {
                success: true,
                scheduleId,
                assignments: assignments.length,
                summary,
                message: `Successfully generated ${assignments.length} call assignments`
            };
            
        } catch (error: any) {
            console.error('❌ Error generating schedule:', error);
            
            // Log error for debugging
            await this.db.collection('errorLogs').add({
                function: 'generateMonthlySchedule',
                error: error.message || 'Unknown error',
                stack: error.stack,
                userId,
                timestamp: admin.firestore.FieldValue.serverTimestamp(),
                data: { month, year, staffingLevel }
            });
            
            return {
                success: false,
                message: `Failed to generate schedule: ${error.message}`
            };
        }
    }
    
    /**
     * Development-phase username validation that accepts all test usernames
     * Updated for 2025-08-16 03:15:44 with username: ramihatou97okay
     */
    private validateUserId(userId: string): void {
        // Add proper sanitization
        if (!userId || typeof userId !== 'string') {
            throw new Error('User ID is required and must be a string');
        }
        const sanitizedId = userId.replace(/[^\w\-\.@]/g, '');
        if (sanitizedId !== userId) {
            throw new Error('Invalid characters in user ID');
        }
        if (!/^[\w\-\.@]{1,128}$/.test(userId)) {
            throw new Error('Invalid user ID format');
        }
        // Log the username for debugging
        console.log(`👤 Processing request for user: ${userId.substring(0, 20)}${userId.length > 20 ? '...' : ''}`);
    }
    
    /**
     * Create safe user ID for database operations
     */
    private getSafeUserId(userId: string): string {
        return userId.substring(0, 50); // Use first 50 chars for safety
    }
    
    /**
     * Save schedule using batched writes to avoid transaction limits
     */
    private async saveScheduleWithBatches(
        month: number,
        year: number,
        staffingLevel: string,
        userId: string,
        assignments: CallAssignment[],
        crossMonthPostCalls: CrossMonthPostCall[],
        summary: any,
        performance: any
    ): Promise<string> {
        // Create schedule document
        const scheduleRef = this.db.collection('monthlySchedules').doc();
        const scheduleId = scheduleRef.id;
        const safeUserId = this.getSafeUserId(userId);
        
        // Create main schedule document
        await scheduleRef.set({
            id: scheduleId,
            month,
            year,
            staffingLevel,
            generatedBy: safeUserId,
            originalUser: userId,
            generatedAt: admin.firestore.FieldValue.serverTimestamp(),
            assignmentCount: assignments.length,
            summary,
            status: 'Active',
            performance,
            crossMonthPostCallCount: crossMonthPostCalls.length
        });
        
        console.log(`📝 Created schedule document with ID: ${scheduleId}`);
        console.log(`Saving ${assignments.length} assignments in batches of ${this.MAX_BATCH_SIZE}`);
        
        // Save assignments in batches
        let currentBatch = this.db.batch();
        let operationCount = 0;
        let batchNumber = 1;
        
        // Process all assignments
        for (let i = 0; i < assignments.length; i++) {
            // Check if we need a new batch
            if (operationCount >= this.MAX_BATCH_SIZE) {
                // Commit current batch with retry logic
                await this.commitBatchWithRetry(currentBatch, `assignments batch ${batchNumber}`);
                console.log(`✅ Committed batch ${batchNumber} with ${operationCount} operations`);
                
                // Start a new batch
                currentBatch = this.db.batch();
                operationCount = 0;
                batchNumber++;
            }
            
            // Add assignment to current batch
            const assignmentRef = this.db.collection('callAssignments').doc();
            currentBatch.set(assignmentRef, {
                ...assignments[i],
                id: assignmentRef.id,
                scheduleId,
                createdBy: safeUserId,
                createdAt: admin.firestore.FieldValue.serverTimestamp()
            });
            
            operationCount++;
        }
        
        // Save cross-month post-calls if any
        if (crossMonthPostCalls.length > 0) {
            const nextMonth = month === 11 ? 0 : month + 1;
            const nextYear = month === 11 ? year + 1 : year;
            
            console.log(`Saving ${crossMonthPostCalls.length} cross-month post-calls for ${nextMonth + 1}/${nextYear}`);
            
            for (let i = 0; i < crossMonthPostCalls.length; i++) {
                // Check if we need a new batch
                if (operationCount >= this.MAX_BATCH_SIZE) {
                    // Commit current batch with retry logic
                    await this.commitBatchWithRetry(currentBatch, `cross-month batch ${batchNumber}`);
                    console.log(`✅ Committed batch ${batchNumber} with ${operationCount} operations`);
                    
                    // Start a new batch
                    currentBatch = this.db.batch();
                    operationCount = 0;
                    batchNumber++;
                }
                
                // Add cross-month post-call to current batch
                const postCallRef = this.db.collection('crossMonthPostCalls').doc();
                currentBatch.set(postCallRef, {
                    ...crossMonthPostCalls[i],
                    id: postCallRef.id,
                    targetMonth: nextMonth,
                    targetYear: nextYear,
                    createdBy: safeUserId,
                    createdAt: admin.firestore.FieldValue.serverTimestamp(),
                    processed: false
                });
                
                operationCount++;
            }
        }

        // Commit final batch if it has any operations
        if (operationCount > 0) {
            await this.commitBatchWithRetry(currentBatch, `final batch ${batchNumber}`);
            console.log(`✅ Committed final batch with ${operationCount} operations`);
        }
        
        return scheduleId;
    }
    
    /**
     * Commit batch with retry logic for reliability
     */
    private async commitBatchWithRetry(
        batch: FirebaseFirestore.WriteBatch, 
        batchLabel: string,
        retryCount = this.MAX_RETRY_ATTEMPTS
    ): Promise<void> {
        try {
            await batch.commit();
        } catch (error: any) {
            console.error(`❌ Error committing ${batchLabel}: ${error.message}`);
            
            if (retryCount > 0) {
                const delay = Math.pow(2, this.MAX_RETRY_ATTEMPTS - retryCount) * 1000; // Exponential backoff
                console.log(`⚠️ Batch commit failed. Retrying in ${delay}ms... (${retryCount} attempts left)`);
                
                // Wait briefly before retry with exponential backoff
                await new Promise(resolve => setTimeout(resolve, delay));
                return this.commitBatchWithRetry(batch, batchLabel, retryCount - 1);
            }
            
            throw new Error(`Failed to commit ${batchLabel} after ${this.MAX_RETRY_ATTEMPTS} attempts: ${error.message}`);
        }
    }
    
    // [Rest of the class implementation remains unchanged]
}