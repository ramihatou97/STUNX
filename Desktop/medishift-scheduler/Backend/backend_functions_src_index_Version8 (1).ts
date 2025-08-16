import * as functions from 'firebase-functions';
import * as admin from 'firebase-admin';
import { MonthlyCallScheduler } from './scheduling/monthly-scheduler';
import { 
    validateRequest, 
    GenerateScheduleRequestSchema,
    rateLimit,
    requireAuth 
} from './middleware/input-validator';

// Initialize Firebase Admin
admin.initializeApp();

const db = admin.firestore();

// Export vacation analyzer
export { analyzeLeaveRequest } from './vacation/vacation-analyzer';

// Generate monthly schedule with validation and rate limiting
export const generateMonthlySchedule = functions
    .runWith({
        timeoutSeconds: 300,
        memory: '2GB'
    })
    .https.onCall(async (data, context) => {
        try {
            // Apply rate limiting
            rateLimit(5, 60000)(context); // 5 requests per minute
            
            // Require authentication
            const auth = requireAuth(context);
            
            // Validate input
            const validatedData = validateRequest(GenerateScheduleRequestSchema)(data, context);
            
            const { month, year, staffingLevel = 'Normal', forceRegenerate = false } = validatedData;
            
            console.log(`📅 Generating schedule for ${month + 1}/${year}`);
            console.log(`👤 Requested by: ${auth.uid}`);
            
            // Check if schedule already exists
            if (!forceRegenerate) {
                const existingSchedule = await db
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
            
            // Get required data with error handling
            const [residentsSnapshot, configSnapshot, academicYearSnapshot, leaveSnapshot] = 
                await Promise.all([
                    db.collection('residents').where('status', '==', 'Active').get(),
                    db.collection('configuration').doc('app').get(),
                    db.collection('academicYears').where('isCurrent', '==', true).limit(1).get(),
                    db.collection('leaveRequests')
                        .where('status', '==', 'Approved')
                        .where('startDate', '<=', admin.firestore.Timestamp.fromDate(
                            new Date(year, month + 1, 0)
                        ))
                        .where('endDate', '>=', admin.firestore.Timestamp.fromDate(
                            new Date(year, month, 1)
                        ))
                        .get()
                ]);
            
            // Validate data exists
            if (residentsSnapshot.empty) {
                throw new functions.https.HttpsError(
                    'failed-precondition',
                    'No active residents found'
                );
            }
            
            if (!configSnapshot.exists) {
                throw new functions.https.HttpsError(
                    'failed-precondition',
                    'Application configuration not found'
                );
            }
            
            if (academicYearSnapshot.empty) {
                throw new functions.https.HttpsError(
                    'failed-precondition',
                    'No current academic year found'
                );
            }
            
            // Prepare data
            const residents = residentsSnapshot.docs.map(doc => ({
                id: doc.id,
                ...doc.data()
            }));
            
            const config = configSnapshot.data();
            const academicYear = academicYearSnapshot.docs[0].data();
            const approvedLeave = leaveSnapshot.docs.map(doc => ({
                id: doc.id,
                ...doc.data()
            }));
            
            // Get existing assignments for the academic year
            const existingAssignments = await db
                .collection('callAssignments')
                .where('date', '>=', admin.firestore.Timestamp.fromDate(
                    new Date(academicYear.startDate.toDate())
                ))
                .where('date', '<', admin.firestore.Timestamp.fromDate(
                    new Date(year, month, 1)
                ))
                .get();
            
            const existing = existingAssignments.docs.map(doc => ({
                id: doc.id,
                ...doc.data()
            }));
            
            // Create scheduler instance
            const scheduler = new MonthlyCallScheduler(
                residents,
                config,
                academicYear,
                approvedLeave,
                month,
                year,
                existing,
                false // Set to true for debug mode
            );
            
            // Generate schedule
            const assignments = await scheduler.generateSchedule(staffingLevel);
            
            // Save to database in batch
            const batch = db.batch();
            const scheduleRef = db.collection('monthlySchedules').doc();
            
            batch.set(scheduleRef, {
                id: scheduleRef.id,
                month,
                year,
                staffingLevel,
                generatedBy: auth.uid,
                generatedAt: admin.firestore.FieldValue.serverTimestamp(),
                assignmentCount: assignments.length,
                summary: scheduler.getScheduleSummary(),
                status: 'Active'
            });
            
            // Save assignments
            assignments.forEach(assignment => {
                const assignmentRef = db.collection('callAssignments').doc();
                batch.set(assignmentRef, {
                    ...assignment,
                    id: assignmentRef.id,
                    scheduleId: scheduleRef.id,
                    createdBy: auth.uid,
                    createdAt: admin.firestore.FieldValue.serverTimestamp()
                });
            });
            
            // Commit batch
            await batch.commit();
            
            console.log(`✅ Schedule generated successfully with ${assignments.length} assignments`);
            
            // Create audit log
            await db.collection('auditLogs').add({
                action: 'generate_monthly_schedule',
                userId: auth.uid,
                timestamp: admin.firestore.FieldValue.serverTimestamp(),
                details: {
                    month,
                    year,
                    staffingLevel,
                    assignmentCount: assignments.length,
                    scheduleId: scheduleRef.id
                }
            });
            
            return {
                success: true,
                scheduleId: scheduleRef.id,
                assignments: assignments.length,
                summary: scheduler.getScheduleSummary(),
                message: `Successfully generated ${assignments.length} call assignments`
            };
            
        } catch (error: any) {
            console.error('❌ Error generating schedule:', error);
            
            // Log error for debugging
            await db.collection('errorLogs').add({
                function: 'generateMonthlySchedule',
                error: error.message || 'Unknown error',
                stack: error.stack,
                userId: context.auth?.uid,
                timestamp: admin.firestore.FieldValue.serverTimestamp(),
                data: { month: data.month, year: data.year }
            });
            
            if (error instanceof functions.https.HttpsError) {
                throw error;
            }
            
            throw new functions.https.HttpsError(
                'internal',
                'Failed to generate schedule',
                { error: error.message }
            );
        }
    });

// Health check endpoint
export const healthCheck = functions.https.onRequest(async (req, res) => {
    try {
        // Check database connection
        await db.collection('_health').doc('check').set({
            timestamp: admin.firestore.FieldValue.serverTimestamp(),
            status: 'ok'
        });
        
        res.status(200).json({
            status: 'healthy',
            timestamp: new Date().toISOString(),
            services: {
                firestore: 'connected',
                functions: 'running'
            }
        });
    } catch (error) {
        res.status(500).json({
            status: 'unhealthy',
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});