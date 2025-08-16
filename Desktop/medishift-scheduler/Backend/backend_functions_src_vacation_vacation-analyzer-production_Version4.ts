import * as functions from 'firebase-functions';
import * as admin from 'firebase-admin';
import sgMail from '@sendgrid/mail';
import Twilio from 'twilio';

// Define types locally since shared module can't be found
interface LeaveRequest {
    id: string;
    residentId: string;
    startDate: Date | admin.firestore.Timestamp;
    endDate: Date | admin.firestore.Timestamp;
    type: string;
    status: string;
    createdAt?: admin.firestore.Timestamp;
    processingId?: string;
    processingStarted?: admin.firestore.Timestamp;
    analysisReportId?: string;
    analysisScore?: number;
    denialJustification?: string;
    updatedAt?: admin.firestore.Timestamp;
    updatedBy?: string;
}

interface Resident {
    id: string;
    vacationDays?: number;
    usedVacationDays?: number;
    email?: string;
    phone?: string;
    pushTokens?: string[];
    role?: string;
}

interface CallAssignment {
    residentId: string;
    date: admin.firestore.Timestamp;
    status: string;
}

interface Notification {
    title: string;
    body: string;
}

const db = admin.firestore();
const TRANSACTION_BATCH_LIMIT = 10; // Firestore has a 500 write limit per transaction
const MAX_LEAVE_QUERY_SIZE = 100; // Limit queries for performance

// Rate limiter to prevent abuse
const rateLimiter = new Map<string, {count: number, reset: number}>();

interface AnalysisResult {
    recommendation: 'Approve' | 'Deny' | 'Review';
    score: number;
    factors: AnalysisFactor[];
    denialReason?: string;
    warnings: string[];
    suggestions: string[];
}

interface AnalysisFactor {
    name: string;
    value: number;
    weight: number;
    description: string;
}

// Wrapper that calls the real implementation defined later in the file.
async function performComprehensiveAnalysis(leaveRequest: LeaveRequest): Promise<AnalysisResult> {
    if (typeof (globalThis as any).performComprehensiveAnalysisImpl === 'function') {
        return await (globalThis as any).performComprehensiveAnalysisImpl(leaveRequest);
    }
    throw new Error('performComprehensiveAnalysis implementation not loaded');
}

// ML-based adjustment for leave analysis scoring
interface MLAdjustment {
    baseScore: number;
    mlAdjustment: number;
    confidence: number;
    factors: string[];
}

// Define HistoricalApproval if not already defined
interface HistoricalApproval {
    score: number;
    approved: boolean;
    factors: string[];
}

// Suggest swap and coverage options for call conflicts
interface ConflictResolution {
    autoSwapPossible: boolean;
    suggestedSwaps: SwapProposal[];
    coverageOptions: CoverageOption[];
}

// Define SwapProposal and CoverageOption if not already defined
interface SwapProposal {
    residentId: string;
    callAssignmentId: string;
    swapWithResidentId: string;
    reason?: string;
}

interface CoverageOption {
    residentId: string;
    callAssignmentId: string;
    coverageType: 'float' | 'backup' | 'volunteer';
    notes?: string;
}

// Exponential backoff retry logic
async function withRetry<T>(
    operation: () => Promise<T>, 
    maxRetries: number = 3, 
    baseDelayMs: number = 200
): Promise<T> {
    let lastError: any;
    
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            return await operation();
        } catch (err: any) {
            // Only retry transient errors
            if (err.code !== 'unavailable' && 
                err.code !== 'resource-exhausted' && 
                err.code !== 'deadline-exceeded') {
                throw err;
            }
            
            lastError = err;
            const delay = baseDelayMs * Math.pow(2, attempt);
            console.log(`Retry attempt ${attempt+1}/${maxRetries} after ${delay}ms`);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
    
    throw lastError;
}

// Circuit breaker for resilient operation execution
class CircuitBreaker {
    private failures = 0;
    private lastFailTime = 0;
    private state: 'CLOSED' | 'OPEN' | 'HALF_OPEN' = 'CLOSED';
    
    async execute<T>(operation: () => Promise<T>): Promise<T> {
        if (this.state === 'OPEN') {
            if (Date.now() - this.lastFailTime > 60000) {
                this.state = 'HALF_OPEN';
            } else {
                throw new Error('Circuit breaker is OPEN');
            }
        }
        
        try {
            const result = await operation();
            if (this.state === 'HALF_OPEN') {
                this.state = 'CLOSED';
                this.failures = 0;
            }
            return result;
        } catch (error) {
            this.failures++;
            this.lastFailTime = Date.now();
            
            if (this.failures >= 5) {
                this.state = 'OPEN';
            }
            throw error;
        }
    }
}

// Primary Cloud Function
export const analyzeLeaveRequest = functions
    .runWith({
        timeoutSeconds: 120,
        memory: '1GB',
        minInstances: 0,
        maxInstances: 10
    })
    .firestore
    .document('leaveRequests/{requestId}')
    .onCreate(async (snap, context) => {
        const leaveRequest = { 
            id: snap.id, 
            ...snap.data() 
        } as LeaveRequest;
        
        const residentId = leaveRequest.residentId;
        const requestId = context.params.requestId;
        
        // Apply rate limiting for abuse prevention
        const key = `${residentId}:analyze`;
        const now = Date.now();
        const limit = rateLimiter.get(key);
        
        if (limit) {
            if (now < limit.reset) {
                if (limit.count >= 5) {
                    console.warn(`Rate limit exceeded for ${residentId}`);
                    await logEvent('rate_limit_exceeded', {
                        residentId,
                        requestId,
                        count: limit.count
                    });
                    return null;
                }
                limit.count++;
            } else {
                rateLimiter.set(key, { count: 1, reset: now + 60000 });
            }
        } else {
            rateLimiter.set(key, { count: 1, reset: now + 60000 });
        }
        
        // Skip if not pending analysis
        if (leaveRequest.status !== 'Pending Analysis') {
            console.log(`Skipping analysis for request ${requestId} with status ${leaveRequest.status}`);
            return null;
        }

        console.log(`🔍 Analyzing leave request ${requestId} for resident ${residentId}`);
        
        // Create an operation ID for tracking
        const operationId = `analyze-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
        
        try {
            // Log start of analysis
            await logEvent('analysis_started', {
                operationId,
                requestId,
                residentId,
                timestamp: admin.firestore.FieldValue.serverTimestamp()
            });
            
            // 1. First check if request document still exists and is still pending
            // This is a lightweight operation before we do heavy processing
            const requestRef = db.collection('leaveRequests').doc(requestId);
            const requestSnapshot = await requestRef.get();
            
            if (!requestSnapshot.exists) {
                console.log(`Request ${requestId} no longer exists`);
                return null;
            }
            
            if (requestSnapshot.data()?.status !== 'Pending Analysis') {
                console.log(`Request ${requestId} already processed, status: ${requestSnapshot.data()?.status}`);
                return null;
            }
            
            // 2. Set a processing flag to prevent parallel processing
            // We update with a small transaction first for quick locking
            try {
                await db.runTransaction(async (tx) => {
                    const doc = await tx.get(requestRef);
                    if (!doc.exists || doc.data()?.status !== 'Pending Analysis') {
                        throw new Error('Request already processed or deleted');
                    }
                    
                    if (doc.data()?.processingId) {
                        throw new Error('Request already being processed');
                    }
                    
                    tx.update(requestRef, {
                        processingId: operationId,
                        processingStarted: admin.firestore.FieldValue.serverTimestamp()
                    });
                });
            } catch (err: any) {
                console.log(`Request ${requestId} already being processed or changed state: ${err.message}`);
                return null;
            }
            
            // Distributed lock to prevent multiple instances from processing the same request
            const lockAcquired = await acquireProcessingLock(requestId, operationId);
            if (!lockAcquired) {
                console.log(`Request ${requestId} is already being processed by another instance`);
                return null;
            }
            
            // 3. Pre-process data and gather information needed for analysis
            // Do this before the main transaction to keep transaction size small
            try {
                // Set a reasonable timeout
                const timeoutPromise = new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('Analysis timed out')), 30000)
                );
                
                const analysisPromise = performComprehensiveAnalysis(leaveRequest);
                
                // Race the analysis against a timeout
                const analysisResult = await Promise.race([
                    analysisPromise,
                    timeoutPromise
                ]) as AnalysisResult;
                
                // 4. Use a transaction for the critical updates
                const result = await withRetry(() => db.runTransaction(async (transaction) => {
                    // Verify request is still valid and being processed by us
                    const latestDoc = await transaction.get(requestRef);
                    
                    if (!latestDoc.exists) {
                        throw new Error('Request document was deleted during analysis');
                    }
                    
                    const currentData = latestDoc.data();
                    if (currentData?.status !== 'Pending Analysis' || 
                        currentData?.processingId !== operationId) {
                        console.log('Request already processed by another instance');
                        return null;
                    }
                    
                    // Create analysis report document
                    const reportRef = db.collection('leaveAnalysisReports').doc();
                    const report = {
                        id: reportRef.id,
                        requestId: requestId,
                        residentId: leaveRequest.residentId,
                        analyzedAt: admin.firestore.Timestamp.now(),
                        operationId,
                        recommendation: analysisResult.recommendation,
                        score: analysisResult.score,
                        factors: analysisResult.factors,
                        warnings: analysisResult.warnings,
                        suggestions: analysisResult.suggestions,
                        denialReason: analysisResult.denialReason,
                        processingTimeMs: Date.now() - currentData?.processingStarted?.toMillis() || 0
                    };
                    
                    // Determine new status
                    const newStatus = analysisResult.recommendation === 'Deny' 
                        ? 'Denied' 
                        : analysisResult.recommendation === 'Review'
                        ? 'Pending Review'
                        : 'Pending Approval';
                    
                    // Update request document
                    transaction.update(requestRef, {
                        status: newStatus,
                        analysisReportId: reportRef.id,
                        analysisScore: analysisResult.score,
                        denialJustification: analysisResult.denialReason,
                        processingId: admin.firestore.FieldValue.delete(),
                        processingStarted: admin.firestore.FieldValue.delete(),
                        updatedAt: admin.firestore.FieldValue.serverTimestamp(),
                        updatedBy: 'system-analyzer'
                    });
                    
                    // Create notification document - inside transaction for atomicity
                    const notificationRef = db.collection('notifications').doc();
                    const notification = {
                        id: notificationRef.id,
                        recipientId: leaveRequest.residentId,
                        type: 'leave_analysis_complete',
                        title: 'Leave Request Analysis Complete',
                        message: `Your leave request for ${formatDate(leaveRequest.startDate)} has been analyzed. Status: ${newStatus}`,
                        data: {
                            requestId: leaveRequest.id,
                            recommendation: analysisResult.recommendation,
                            status: newStatus
                        },
                        read: false,
                        createdAt: admin.firestore.FieldValue.serverTimestamp()
                    };
                    
                    // Store both documents in transaction
                    transaction.set(reportRef, report);
                    transaction.set(notificationRef, notification);
                    
                    // Return the result
                    return { 
                        reportId: reportRef.id,
                        recommendation: analysisResult.recommendation,
                        status: newStatus,
                        notificationId: notificationRef.id
                    };
                }));
                
                // 5. Post-transaction processing
                if (result) {
                    const resultAny = result as any;
                    // Log success
                    await logEvent('analysis_complete', {
                        operationId,
                        requestId,
                        result: resultAny,
                        timestamp: admin.firestore.FieldValue.serverTimestamp()
                    });

                    // Send additional notifications outside transaction (non-critical)
                    if (resultAny.recommendation === 'Review') {
                        await notifyChiefResidents(leaveRequest, resultAny.status);
                    }

                    console.log(`✅ Analysis complete: ${resultAny.recommendation} for request ${requestId}`);
                }
                
                return result;
                
            } catch (error: any) {
                // If we get here, we failed even after retries
                console.error(`❌ Fatal error analyzing leave request:`, error);
                
                // Update request with error status - but don't fail if this update fails
                try {
                    await requestRef.update({
                        status: 'Analysis Failed',
                        error: error.message || 'Unknown error occurred',
                        processingId: admin.firestore.FieldValue.delete(),
                        processingStarted: admin.firestore.FieldValue.delete(),
                        updatedAt: admin.firestore.FieldValue.serverTimestamp()
                    });
                } catch (updateError) {
                    console.error('Failed to update error status:', updateError);
                }
                
                // Create error notification
                try {
                    await createErrorNotification(leaveRequest, error.message);
                } catch (notifError) {
                    console.error('Failed to send error notification:', notifError);
                }
                
                // Log the failure for monitoring
                await logEvent('analysis_failed', {
                    operationId,
                    requestId,
                    error: error.message,
                    stack: error.stack,
                    timestamp: admin.firestore.FieldValue.serverTimestamp()
                });
                
                throw new functions.https.HttpsError(
                    'internal',
                    `Failed to analyze leave request: ${error.message}`,
                    { requestId: leaveRequest.id }
                );
            }
            
        } catch (error: any) {
            console.error(`Unhandled error in analyzeLeaveRequest:`, error);
            
            // Final fallback error logging
            try {
                await db.collection('errorLogs').add({
                    function: 'analyzeLeaveRequest',
                    requestId,
                    operationId,
                    error: error.message,
                    stack: error.stack,
                    timestamp: admin.firestore.FieldValue.serverTimestamp()
                });
            } catch (logError) {
                console.error('Failed to log error:', logError);
            }
            
            throw error;
        }
    });

// Optional: Batch analysis of leave requests (scheduled)
export const batchAnalyzeLeaveRequests = functions
    .runWith({ memory: '2GB', timeoutSeconds: 540 })
    .pubsub.schedule('every 5 minutes')
    .onRun(async () => {
        const pending = await db.collection('leaveRequests')
            .where('status', '==', 'Pending Analysis')
            .where('createdAt', '<', admin.firestore.Timestamp.fromMillis(
                Date.now() - 60000 // At least 1 minute old
            ))
            .limit(10)
            .get();
        
        const batch = db.batch();
        const analyses = await Promise.allSettled(
            pending.docs.map(doc => analyzeRequest(doc))
        );
        
        // Process results, e.g., commit batch updates, log outcomes, etc.
        // ...existing code or custom logic...
    });

// Replace the previously declared analyzeRequest stub with a working implementation
async function analyzeRequest(doc: admin.firestore.QueryDocumentSnapshot): Promise<void> {
    const leaveRequest = { id: doc.id, ...doc.data() } as LeaveRequest;
    const requestRef = db.collection('leaveRequests').doc(doc.id);
    const operationId = `batch-analyze-${Date.now()}-${Math.random().toString(36).slice(2,9)}`;

    try {
        // Lightweight guard: only analyze pending items
        if (leaveRequest.status !== 'Pending Analysis') return;

        await logEvent('batch_analysis_started', { operationId, requestId: leaveRequest.id, residentId: leaveRequest.residentId, timestamp: admin.firestore.FieldValue.serverTimestamp() });

        // Acquire processing lock to avoid duplicates
        const locked = await acquireProcessingLock(leaveRequest.id, operationId);
        if (!locked) {
            console.log(`Could not acquire lock for ${leaveRequest.id}`);
            return;
        }

        const analysisResult = await performComprehensiveAnalysis(leaveRequest);

        // Minimal transactional update: write report and update request
        await db.runTransaction(async (tx) => {
            const requestSnap = await tx.get(requestRef);
            if (!requestSnap.exists) return;
            const current = requestSnap.data();
            if (current?.status !== 'Pending Analysis') return;

            const reportRef = db.collection('leaveAnalysisReports').doc();
            const report = {
                id: reportRef.id,
                requestId: leaveRequest.id,
                residentId: leaveRequest.residentId,
                analyzedAt: admin.firestore.Timestamp.now(),
                operationId,
                recommendation: analysisResult.recommendation,
                score: analysisResult.score,
                factors: analysisResult.factors,
                warnings: analysisResult.warnings,
                suggestions: analysisResult.suggestions,
                denialReason: analysisResult.denialReason
            };

            const newStatus = analysisResult.recommendation === 'Deny' ? 'Denied' : analysisResult.recommendation === 'Review' ? 'Pending Review' : 'Pending Approval';

            tx.set(reportRef, report);
            tx.update(requestRef, {
                status: newStatus,
                analysisReportId: reportRef.id,
                analysisScore: analysisResult.score,
                denialJustification: analysisResult.denialReason,
                updatedAt: admin.firestore.FieldValue.serverTimestamp()
            });

            // Create a simple notification record
            const notificationRef = db.collection('notifications').doc();
            tx.set(notificationRef, {
                id: notificationRef.id,
                recipientId: leaveRequest.residentId,
                type: 'leave_analysis_complete',
                title: 'Leave Request Analysis Complete',
                message: `Your leave request for ${formatDate(leaveRequest.startDate)} has been analyzed. Status: ${newStatus}`,
                data: { requestId: leaveRequest.id, recommendation: analysisResult.recommendation, status: newStatus },
                read: false,
                createdAt: admin.firestore.FieldValue.serverTimestamp()
            });
        });

        await logEvent('batch_analysis_complete', { operationId, requestId: leaveRequest.id, timestamp: admin.firestore.FieldValue.serverTimestamp() });

    } catch (err: any) {
        console.error('analyzeRequest error', err);
        try {
            await requestRef.update({ status: 'Analysis Failed', error: err.message, updatedAt: admin.firestore.FieldValue.serverTimestamp() });
        } catch (e) {
            console.warn('Failed to mark request as failed', e);
        }
        await logEvent('batch_analysis_failed', { operationId, requestId: leaveRequest.id, error: err.message, timestamp: admin.firestore.FieldValue.serverTimestamp() });
    } finally {
        // Best-effort: release any locks
        try { await releaseProcessingLock(leaveRequest.id, operationId); } catch (e) { /* ignore */ }
    }
}

// --- Helper implementations for missing utilities ---

async function logEvent(eventName: string, payload: Record<string, any>): Promise<void> {
    try {
        await db.collection('functionLogs').add({ event: eventName, payload, timestamp: admin.firestore.FieldValue.serverTimestamp() });
    } catch (e) {
        console.warn('logEvent failed', e);
    }
}

function formatDate(input: any): string {
    try {
        const d = input && typeof input.toDate === 'function' ? input.toDate() : new Date(input);
        return d.toISOString().split('T')[0]; // YYYY-MM-DD
    } catch (e) {
        return String(input);
    }
}

function generateDenialReason(factors: AnalysisFactor[], warnings: string[]): string {
    // Simple heuristic: pick top negative factor + first warning
    const negative = factors.slice().sort((a, b) => (a.value * a.weight) - (b.value * b.weight)).slice(0, 2);
    const parts: string[] = [];
    if (negative.length) parts.push(`Concerns: ${negative.map(n => n.name).join(', ')}`);
    if (warnings.length) parts.push(warnings[0]);
    return parts.join(' - ') || 'Denied due to policy constraints';
}

// Processing lock using a lightweight Firestore document
async function acquireProcessingLock(requestId: string, operationId: string): Promise<boolean> {
    const lockRef = db.collection('processingLocks').doc(requestId);
    try {
        await db.runTransaction(async (tx) => {
            const lockSnap = await tx.get(lockRef);
            if (lockSnap.exists && lockSnap.data()?.operationId) {
                throw new Error('locked');
            }
            tx.set(lockRef, { operationId, acquiredAt: admin.firestore.FieldValue.serverTimestamp() });
        });
        return true;
    } catch (e) {
        return false;
    }
}

async function releaseProcessingLock(requestId: string, operationId: string): Promise<void> {
    const lockRef = db.collection('processingLocks').doc(requestId);
    try {
        const doc = await lockRef.get();
        if (doc.exists && doc.data()?.operationId === operationId) {
            await lockRef.delete();
        }
    } catch (e) {
        // ignore
    }
}

function normalizeDate(d: Date): Date {
    // Normalize to UTC midnight to avoid timezone shifts
    const dt = new Date(d);
    return new Date(Date.UTC(dt.getUTCFullYear(), dt.getUTCMonth(), dt.getUTCDate(), 0, 0, 0, 0));
}

function checkAdvanceNotice(startDate: Date): AnalysisFactor {
    const now = new Date();
    const days = Math.ceil((startDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24));
    let value = 1.0;
    if (days < 14) value = 0.2;
    else if (days < 30) value = 0.6;
    return { name: 'Advance Notice', value, weight: 1, description: `${days} days notice` };
}

async function checkPeakPeriodConflicts(startDate: Date, endDate: Date): Promise<AnalysisFactor> {
    // Placeholder: check if dates overlap known peak periods (holidays, exams)
    const peak = false; // replace with real logic
    return { name: 'Peak Period', value: peak ? 0.2 : 1.0, weight: 2, description: peak ? 'Overlaps with peak period' : 'No peak overlap' };
}

async function checkRecentLeaveHistory(residentId: string, startDate: Date): Promise<AnalysisFactor> {
    // Check past 90 days for leaves
    const cutoff = admin.firestore.Timestamp.fromMillis(Date.now() - 90 * 24 * 60 * 60 * 1000);
    const q = await db.collection('leaveRequests').where('residentId', '==', residentId).where('createdAt', '>=', cutoff).get();
    const recent = q.size;
    const value = recent > 3 ? 0.2 : recent > 0 ? 0.6 : 1.0;
    return { name: 'Recent Leave History', value, weight: 1, description: `${recent} requests in last 90 days` };
}

async function fetchHistoricalApprovals(residentId: string): Promise<HistoricalApproval[]> {
    // Fetch recent analysis reports for the resident
    const snaps = await db.collection('leaveAnalysisReports').where('residentId', '==', residentId).orderBy('analyzedAt', 'desc').limit(50).get();
    return snaps.docs.map(d => ({ score: d.data().score || 0, approved: d.data().recommendation === 'Approve', factors: d.data().factors || [] }));
}

// Simple notify functions
async function notifyChiefResidents(leaveRequest: LeaveRequest, status: string): Promise<void> {
    // Find chief residents and create notifications (placeholder)
    const chiefs = await db.collection('residents').where('role', '==', 'chief').get();
    const batch = db.batch();
    chiefs.docs.forEach(c => {
        const n = db.collection('notifications').doc();
        batch.set(n, {
            id: n.id,
            recipientId: c.id,
            type: 'leave_review_needed',
            title: 'Leave Review Required',
            message: `Leave request ${leaveRequest.id} requires review. Status: ${status}`,
            data: { requestId: leaveRequest.id },
            read: false,
            createdAt: admin.firestore.FieldValue.serverTimestamp()
        });
    });
    await batch.commit();
}

async function createErrorNotification(leaveRequest: LeaveRequest, errorMessage: string): Promise<void> {
    try {
        await db.collection('notifications').add({
            recipientId: leaveRequest.residentId,
            type: 'leave_analysis_error',
            title: 'Leave Analysis Error',
            message: `An error occurred while analyzing your leave request: ${errorMessage}`,
            data: { requestId: leaveRequest.id },
            read: false,
            createdAt: admin.firestore.FieldValue.serverTimestamp()
        });
    } catch (e) {
        console.warn('createErrorNotification failed', e);
    }
}

// Analytics data model and generators
interface AnalyticsData {
    approvalRate: number; // 0..1
    averageProcessingTime: number; // milliseconds
    denialReasons: Record<string, number>;
    peakRequestTimes: Date[]; // top moments for requests
    coverageImpact: number; // heuristic 0..1 where 1 is high impact
}

export async function generateAnalytics(
    period: { start: Date; end: Date }
): Promise<AnalyticsData> {
    const requests = await getRequestsInPeriod(period);

    const approvalRate = calculateApprovalRate(requests);
    const averageProcessingTime = calculateAvgProcessingTime(requests);
    const denialReasons = aggregateDenialReasons(requests);
    const peakRequestTimes = identifyPeakTimes(requests);
    const coverageImpact = assessCoverageImpact(requests, peakRequestTimes);

    return {
        approvalRate,
        averageProcessingTime,
        denialReasons,
        peakRequestTimes,
        coverageImpact,
    };
}

// Fetch leave requests in the given period with pagination
async function getRequestsInPeriod(period: { start: Date; end: Date }): Promise<any[]> {
    const startTs = admin.firestore.Timestamp.fromDate(period.start);
    const endTs = admin.firestore.Timestamp.fromDate(period.end);

    const results: any[] = [];
    let query = db.collection('leaveRequests')
         .where('createdAt', '>=', startTs)
         .where('createdAt', '<=', endTs)
         .orderBy('createdAt')
         .limit(500);

    let lastDoc: admin.firestore.QueryDocumentSnapshot | undefined;
    while (true) {
        let q = query;
        if (lastDoc) q = q.startAfter(lastDoc);
        const snap = await q.get();
        if (snap.empty) break;
        snap.forEach(d => results.push({ id: d.id, ...d.data() }));
        lastDoc = snap.docs[snap.docs.length - 1];
        if (snap.size < 500) break;
    }

    return results;
}

function calculateApprovalRate(requests: any[]): number {
    if (!requests.length) return 0;
    const approved = requests.filter(r => r.status === 'Approved').length;
    return approved / requests.length;
}

function calculateAvgProcessingTime(requests: any[]): number {
    const times: number[] = [];
    for (const r of requests) {
        if (r.processedAt && r.createdAt) {
            const processed = r.processedAt instanceof admin.firestore.Timestamp
                ? r.processedAt.toDate()
                : new Date(r.processedAt);
            const created = r.createdAt instanceof admin.firestore.Timestamp
                ? r.createdAt.toDate()
                : new Date(r.createdAt);
            times.push(processed.getTime() - created.getTime());
        }
    }
    if (!times.length) return 0;
    const sum = times.reduce((s, v) => s + v, 0);
    return Math.round(sum / times.length);
}

function aggregateDenialReasons(requests: any[]): Record<string, number> {
    const counts: Record<string, number> = {};
    for (const r of requests) {
        if (r.status === 'Denied') {
            const reason = r.denialReason || 'unspecified';
            counts[reason] = (counts[reason] || 0) + 1;
        }
    }
    return counts;
}

function identifyPeakTimes(requests: any[]): Date[] {
    // Group by hour window and pick top 3
    const buckets: Record<string, number> = {};
    for (const r of requests) {
        const created = r.createdAt instanceof admin.firestore.Timestamp
            ? r.createdAt.toDate()
            : new Date(r.createdAt);
        const key = `${created.getFullYear()}-${created.getMonth()+1}-${created.getDate()}T${created.getHours()}`;
        buckets[key] = (buckets[key] || 0) + 1;
    }

    const sorted = Object.entries(buckets).sort((a, b) => b[1] - a[1]).slice(0, 3);
    return sorted.map(([k]) => {
        // parse back into Date at start of hour
        const [datePart, hourPart] = k.split('T');
        const [y, m, d] = datePart.split('-').map(Number);
        const hour = Number(hourPart);
        return new Date(y, m - 1, d, hour, 0, 0, 0);
    });
}

function assessCoverageImpact(requests: any[], peaks: Date[]): number {
    // Simple heuristic: fraction of requests that fall into the top peak hours
    if (!requests.length || !peaks.length) return 0;
    const peakSet = new Set(peaks.map(p => p.getTime()));
    let peakCount = 0;
    for (const r of requests) {
        const created = r.createdAt instanceof admin.firestore.Timestamp
            ? r.createdAt.toDate()
            : new Date(r.createdAt);
        const hourStart = new Date(created.getFullYear(), created.getMonth(), created.getDate(), created.getHours(), 0, 0, 0).getTime();
        if (peakSet.has(hourStart)) peakCount++;
    }
    return +(peakCount / requests.length).toFixed(3); // 0..1
}

// Concrete implementation of the comprehensive analysis
async function performComprehensiveAnalysisImpl(leaveRequest: LeaveRequest): Promise<AnalysisResult> {
    const factors: AnalysisFactor[] = [];
    const warnings: string[] = [];
    const suggestions: string[] = [];

    // Basic validation
    if (!leaveRequest || !leaveRequest.residentId || !leaveRequest.startDate || !leaveRequest.endDate) {
        throw new Error('Invalid leave request data');
    }

    // Fetch resident
    const residentSnap = await db.collection('residents').doc(leaveRequest.residentId).get();
    if (!residentSnap.exists) throw new Error('Resident not found');
    const resident = residentSnap.data() as Resident;

    // Normalize dates
    const startDate = leaveRequest.startDate && typeof (leaveRequest.startDate as any).toDate === 'function'
        ? normalizeDate((leaveRequest.startDate as any).toDate())
        : normalizeDate(new Date(leaveRequest.startDate as any));
    const endDate = leaveRequest.endDate && typeof (leaveRequest.endDate as any).toDate === 'function'
        ? normalizeDate((leaveRequest.endDate as any).toDate())
        : normalizeDate(new Date(leaveRequest.endDate as any));

    if (startDate > endDate) throw new Error('Start date after end date');

    const durationDays = Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24)) + 1;

    // Compose factors using existing helpers
    const balanceFactor = await checkVacationBalance(resident, durationDays, leaveRequest.type);
    factors.push(balanceFactor);
    if (balanceFactor.value < 0.5) warnings.push('Insufficient vacation balance');

    const coverageFactor = await checkCoverageAvailability(startDate, endDate, resident);
    factors.push(coverageFactor);
    if (coverageFactor.value < 0.4) warnings.push('Low coverage during requested dates');

    const conflictFactor = await checkCallConflicts(leaveRequest.residentId, startDate, endDate);
    factors.push(conflictFactor);
    if (conflictFactor.value < 1.0) {
        warnings.push('Conflicts with scheduled calls');
        suggestions.push('Consider arranging swaps or coverage');
    }

    const noticeFactor = checkAdvanceNotice(startDate);
    factors.push(noticeFactor);
    if (noticeFactor.value < 0.5) warnings.push('Short notice for leave');

    const peakFactor = await checkPeakPeriodConflicts(startDate, endDate);
    factors.push(peakFactor);
    if (peakFactor.value < 0.7) warnings.push('Overlaps with peak period');

    const historyFactor = await checkRecentLeaveHistory(leaveRequest.residentId, startDate);
    factors.push(historyFactor);
    if (historyFactor.value < 0.5) warnings.push('Recent leave history may affect approval');

    // Weighted score
    const totalWeight = factors.reduce((s, f) => s + (f.weight || 0), 0) || 1;
    const weightedScore = factors.reduce((s, f) => s + (f.value * (f.weight || 1)), 0) / totalWeight;

    // Initial recommendation
    let recommendation: 'Approve' | 'Deny' | 'Review';
    let denialReason: string | undefined;

    if (weightedScore >= 0.7) recommendation = 'Approve';
    else if (weightedScore >= 0.4) { recommendation = 'Review'; suggestions.push('Manual review recommended'); }
    else { recommendation = 'Deny'; denialReason = generateDenialReason(factors, warnings); }

    // ML adjustment
    const historicalData = await fetchHistoricalApprovals(leaveRequest.residentId);
    const mlAdj = await applyMLScoring({ recommendation, score: weightedScore, factors, warnings, suggestions, denialReason } as any, historicalData);
    const finalScore = Math.max(0, Math.min(1, weightedScore + (mlAdj.mlAdjustment || 0)));

    return {
        recommendation,
        score: finalScore,
        factors,
        denialReason,
        warnings,
        suggestions
    };
}

// Expose implementation to wrapper
(globalThis as any).performComprehensiveAnalysisImpl = performComprehensiveAnalysisImpl;

// Module-level circuit breaker for notification delivery
const notificationCircuit = new CircuitBreaker();

// Scheduled processor to deliver scheduledNotifications
export const processScheduledNotifications = functions
    .runWith({ memory: '512MB', timeoutSeconds: 120 })
    .pubsub.schedule('every 5 minutes')
    .onRun(async () => {
        const nowTs = admin.firestore.Timestamp.now();

        // Fetch a batch of pending scheduled notifications (delivered=false)
        const snaps = await db.collection('scheduledNotifications')
            .where('delivered', '==', false)
            .limit(100)
            .get();

        if (snaps.empty) return null;

        const dueDocs = snaps.docs.filter(d => {
            const data = d.data();
            const sendAt: admin.firestore.Timestamp | undefined = data.sendAt;
            const nextAttemptAt: admin.firestore.Timestamp | undefined = data.nextAttemptAt;
            const sendDue = sendAt && sendAt.toMillis() <= nowTs.toMillis();
            const retryDue = nextAttemptAt && nextAttemptAt.toMillis() <= nowTs.toMillis();
            return !!(sendDue || retryDue || (!sendAt && !nextAttemptAt));
        });

        await Promise.all(dueDocs.map(async (doc) => {
            const id = doc.id;
            const data = doc.data();
            const recipientId: string = data.recipientId;
            const notification = data.notification as Notification;
            const attempts: number = data.attempts || 0;

            const operationId = `notify-${Date.now()}-${Math.random().toString(36).slice(2,8)}`;

            try {
                await auditLog({ timestamp: admin.firestore.Timestamp.now(), action: 'scheduled_notification_attempt', actor: 'system-scheduler', target: id, changes: { attempts, recipientId }, metadata: { functionVersion: 'v1' } });

                // Fetch user prefs and attempt delivery through circuit breaker
                const prefs = await getUserPreferences(recipientId);

                await notificationCircuit.execute(async () => {
                    // Use direct channel delivery to honor scheduled timing
                    await sendViaPreferredChannels(recipientId, notification, prefs);
                });

                // Mark as delivered
                await db.collection('scheduledNotifications').doc(id).update({
                    delivered: true,
                    deliveredAt: admin.firestore.FieldValue.serverTimestamp(),
                    attempts: attempts + 1
                });

                await auditLog({ timestamp: admin.firestore.Timestamp.now(), action: 'scheduled_notification_delivered', actor: 'system-scheduler', target: id, changes: { delivered: true }, metadata: { functionVersion: 'v1' } });

            } catch (err: any) {
                const nextAttempts = attempts + 1;
                const backoffMs = Math.min(60 * 60 * 1000, Math.pow(2, attempts) * 60 * 1000); // up to 1 hour
                const nextAttemptAt = admin.firestore.Timestamp.fromMillis(Date.now() + backoffMs);

                const update: Record<string, any> = {
                    attempts: nextAttempts,
                    lastError: String(err?.message || err),
                    lastAttemptAt: admin.firestore.FieldValue.serverTimestamp(),
                    nextAttemptAt
                };

                if (nextAttempts >= 5) {
                    update.failed = true;
                }

                try {
                    await db.collection('scheduledNotifications').doc(id).update(update);
                } catch (e) {
                    console.warn('Failed to update scheduledNotification retry state', e);
                }

                await auditLog({ timestamp: admin.firestore.Timestamp.now(), action: 'scheduled_notification_failed', actor: 'system-scheduler', target: id, changes: { attempts: nextAttempts, error: String(err?.message || err) }, metadata: { functionVersion: 'v1' } });
            }
        }));

        return null;
    });

// Audit logging for compliance and traceability (ensure defined for scheduler)
interface AuditEntry {
    timestamp: admin.firestore.Timestamp;
    action: string;
    actor: string;
    target: string;
    changes: Record<string, any>;
    metadata?: Record<string, any>;
}

async function auditLog(entry: AuditEntry): Promise<void> {
    try {
        await db.collection('auditLogs').add({ ...entry, createdAt: admin.firestore.FieldValue.serverTimestamp() });
    } catch (e) {
        console.warn('auditLog failed', e);
    }
}

// Notification preferences & helpers (ensure present for scheduler)
interface NotificationPreferences {
    email: boolean;
    push: boolean;
    sms: boolean;
    quietHours: { start: string; end: string };
    urgentOnly: boolean;
}

async function getUserPreferences(recipientId: string): Promise<NotificationPreferences> {
    try {
        const doc = await db.collection('userPreferences').doc(recipientId).get();
        if (doc.exists) return doc.data() as NotificationPreferences;
    } catch (e) {
        console.warn('getUserPreferences failed', e);
    }
    return { email: true, push: true, sms: false, quietHours: { start: '22:00', end: '07:00' }, urgentOnly: false };
}

async function sendViaPreferredChannels(recipientId: string, notification: Notification, prefs: NotificationPreferences): Promise<void> {
    const tasks: Promise<void>[] = [];
    if (prefs.email) tasks.push(sendEmailNotification(recipientId, notification));
    if (prefs.push) tasks.push(sendPushNotification(recipientId, notification));
    if (prefs.sms) tasks.push(sendSmsNotification(recipientId, notification));
    await Promise.allSettled(tasks);
}

async function sendEmailNotification(recipientId: string, notification: Notification): Promise<void> {
    const sgKey = process.env.SENDGRID_API_KEY || '';
    if (!sgKey) {
        console.warn('SendGrid API key not configured, skipping email send');
        return;
    }
    try {
        sgMail.setApiKey(sgKey);
        const contact = await getContactInfo(recipientId);
        if (!contact?.email) {
            console.warn('No email for recipient', recipientId);
            return;
        }

        const msg = {
            to: contact.email,
            from: process.env.DEFAULT_FROM_EMAIL || 'no-reply@medishift.example',
            subject: notification.title,
            text: notification.body,
            html: notification.body
        } as any;

        await sgMail.send(msg);
        console.log('Email sent to', contact.email);
    } catch (e) {
        console.error('sendEmailNotification failed', e);
    }
}

async function sendPushNotification(recipientId: string, notification: Notification): Promise<void> {
    try {
        const contact = await getContactInfo(recipientId);
        const tokens: string[] = contact?.pushTokens || [];
        if (!tokens.length) {
            console.warn('No push tokens for recipient', recipientId);
            return;
        }

        const message = {
            notification: {
                title: notification.title,
                body: notification.body
            },
            tokens
        } as admin.messaging.MulticastMessage;

        const resp = await admin.messaging().sendMulticast(message);
        if (resp.failureCount && resp.responses) {
            resp.responses.forEach((r, i) => {
                if (!r.success) console.warn('Push send failed for token', tokens[i], r.error);
            });
        }
        console.log('Push sent to', tokens.length, 'tokens');
    } catch (e) {
        console.error('sendPushNotification failed', e);
    }
}

async function sendSmsNotification(recipientId: string, notification: Notification): Promise<void> {
    const sid = process.env.TWILIO_ACCOUNT_SID || '';
    const token = process.env.TWILIO_AUTH_TOKEN || '';
    const fromNumber = process.env.TWILIO_FROM_NUMBER || '';

    if (!sid || !token || !fromNumber) {
        console.warn('Twilio not configured, skipping SMS');
        return;
    }

    try {
        const client = Twilio(sid, token);
        const contact = await getContactInfo(recipientId);
        if (!contact?.phone) {
            console.warn('No phone number for recipient', recipientId);
            return;
        }

        const message = await client.messages.create({
            body: notification.body,
            from: fromNumber,
            to: contact.phone
        });

        console.log('SMS sent, sid=', message.sid);
    } catch (e) {
        console.error('sendSmsNotification failed', e);
    }
}

// Helper: fetch contact info (email, phone, push tokens) from residents or users collection
async function getContactInfo(recipientId: string): Promise<{ email?: string; phone?: string; pushTokens?: string[] } | null> {
    try {
        const r = await db.collection('residents').doc(recipientId).get();
        if (r.exists) {
            const d = r.data() as any;
            return { email: d.email, phone: d.phone, pushTokens: d.pushTokens || [] };
        }

        const u = await db.collection('users').doc(recipientId).get();
        if (u.exists) {
            const d = u.data() as any;
            return { email: d.email, phone: d.phone, pushTokens: d.pushTokens || [] };
        }
    } catch (e) {
        console.warn('getContactInfo failed', e);
    }
    return null;

}

// Fallback/simple implementations for helpers referenced by the analysis implementation
async function checkVacationBalance(
    resident: Resident,
    requestedDays: number,
    leaveType: string
): Promise<AnalysisFactor> {
    const totalAllowance = resident.vacationDays ?? 20;
    const used = resident.usedVacationDays ?? 0;
    const remaining = Math.max(0, totalAllowance - used);
    const isEmergency = leaveType === 'Emergency' || leaveType === 'Bereavement';
    const value = isEmergency ? 0.9 : (remaining >= requestedDays ? 1.0 : Math.max(0, remaining / requestedDays));
    return { name: 'Vacation Balance', value, weight: isEmergency ? 1 : 3, description: `${remaining} days remaining` };
}
async function checkCoverageAvailability(startDate: Date, endDate: Date, resident: Resident): Promise<AnalysisFactor> {
    try {
        const overlapping = await db.collection('leaveRequests')
            .where('status', 'in', ['Approved', 'Pending Approval'])
            .where('startDate', '<=', admin.firestore.Timestamp.fromDate(endDate))
            .where('endDate', '>=', admin.firestore.Timestamp.fromDate(startDate))
            .limit(50)
            .get();

        // simplistic: if many overlaps, low availability
        const overlaps = overlapping.docs.filter(d => d.data().residentId !== resident.id).length;
        const value = Math.max(0, 1 - overlaps / 5); // degrade as overlaps increase
        return { name: 'Coverage Availability', value, weight: 2, description: `${overlaps} overlapping leaves` };
    } catch (e) {
        return { name: 'Coverage Availability', value: 0.5, weight: 2, description: 'Unable to determine coverage' };
    }
}

async function checkCallConflicts(residentId: string, startDate: Date, endDate: Date): Promise<AnalysisFactor> {
    try {
        const snaps = await db.collection('callAssignments')
            .where('residentId', '==', residentId)
            .where('status', '==', 'Scheduled')
            .where('date', '>=', admin.firestore.Timestamp.fromDate(startDate))
            .limit(20)
            .get();
        const conflicts = snaps.docs.filter(d => d.data().date.toDate() <= endDate).length;
        const value = conflicts === 0 ? 1.0 : 0.0;
        return { name: 'Call Conflicts', value, weight: 3, description: conflicts === 0 ? 'No conflicts' : `${conflicts} conflicts` };
    } catch (e) {
        return { name: 'Call Conflicts', value: 0.5, weight: 3, description: 'Unable to check call conflicts' };
    }
}

async function applyMLScoring(analysisResult: any, historicalData: HistoricalApproval[]): Promise<MLAdjustment> {
    try {
        // Simple heuristic: if historical approval rate high, slightly boost
        const approved = historicalData.filter(h => h.approved).length;
        const rate = historicalData.length ? approved / historicalData.length : 0.5;
        const adjustment = rate > 0.7 ? 0.05 : rate < 0.3 ? -0.05 : 0;
        return { baseScore: analysisResult.score ?? 0, mlAdjustment: adjustment, confidence: 0.6, factors: ['historicalApprovalRate'] };
    } catch (e) {
        return { baseScore: analysisResult.score ?? 0, mlAdjustment: 0, confidence: 0, factors: [] };
    }
}
