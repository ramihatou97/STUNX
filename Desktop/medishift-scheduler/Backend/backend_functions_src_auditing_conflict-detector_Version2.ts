import * as functions from 'firebase-functions';
import * as admin from 'firebase-admin';
import { 
    LeaveRequest, 
    CallAssignment, 
    ORAssignment, 
    ClinicAssignment, 
    ConflictResolutionTicket 
} from '../../../shared/types';

const db = admin.firestore();

interface ConflictingAssignment {
    type: 'Call' | 'OR' | 'Clinic';
    description: string;
    date: admin.firestore.Timestamp;
}

/**
 * Nightly Conflict Auditor - Runs every 24 hours
 * Audits for conflicts between approved leave and finalized schedules
 */
export const nightlyConflictAudit = functions.pubsub
    .schedule('every 24 hours')
    .timeZone('America/New_York')
    .onRun(async (context) => {
        console.log('🤖 Starting Nightly Conflict Audit...');
        
        try {
            const approvedLeaveSnapshot = await db.collection('leaveRequests')
                .where('status', '==', 'Approved')
                .where('startDate', '>=', admin.firestore.Timestamp.now())
                .get();

            if (approvedLeaveSnapshot.empty) {
                console.log('No upcoming approved leave to audit. Exiting.');
                return null;
            }

            const auditPromises = approvedLeaveSnapshot.docs.map(doc => {
                const leaveRequest = { 
                    id: doc.id, 
                    ...doc.data() 
                } as LeaveRequest;
                return findAndFlagConflicts(leaveRequest);
            });

            await Promise.all(auditPromises);
            
            console.log('✅ Nightly Conflict Audit Complete.');
            return null;
        } catch (error) {
            console.error('❌ Conflict audit failed:', error);
            throw error;
        }
    });

async function findAndFlagConflicts(leaveRequest: LeaveRequest): Promise<void> {
    const conflictingAssignments: ConflictingAssignment[] = [];

    try {
        // Query all relevant schedules for the resident during their leave period
        const [callAssignments, orAssignments, clinicAssignments] = await Promise.all([
            getAssignmentsForPeriod('monthlySchedules', leaveRequest),
            getAssignmentsForPeriod('weeklySchedules', leaveRequest, 'or'),
            getAssignmentsForPeriod('weeklySchedules', leaveRequest, 'clinic')
        ]);

        // Process call assignments
        if (callAssignments.length > 0) {
            conflictingAssignments.push(
                ...callAssignments.map((a: CallAssignment) => ({
                    type: 'Call' as const,
                    description: a.type,
                    date: a.date
                }))
            );
        }

        // Process OR assignments
        if (orAssignments.length > 0) {
            conflictingAssignments.push(
                ...orAssignments.map((a: ORAssignment) => ({
                    type: 'OR' as const,
                    description: a.caseType || 'OR Assignment',
                    date: a.date
                }))
            );
        }

        // Process clinic assignments
        if (clinicAssignments.length > 0) {
            conflictingAssignments.push(
                ...clinicAssignments.map((a: ClinicAssignment) => ({
                    type: 'Clinic' as const,
                    description: a.clinicType || 'Clinic',
                    date: a.date
                }))
            );
        }

        // Create conflict ticket if conflicts found
        if (conflictingAssignments.length > 0) {
            console.warn(`🚨 CONFLICT DETECTED for Leave Request: ${leaveRequest.id}`);
            
            const ticketRef = db.collection('conflictTickets').doc();
            const newTicket: ConflictResolutionTicket = {
                id: ticketRef.id,
                leaveRequestId: leaveRequest.id,
                residentId: leaveRequest.residentId,
                residentName: leaveRequest.residentName,
                conflictStartDate: leaveRequest.startDate,
                conflictEndDate: leaveRequest.endDate,
                conflictingAssignments,
                status: 'Open',
                createdAt: admin.firestore.Timestamp.now(),
                priority: calculatePriority(conflictingAssignments)
            };

            const batch = db.batch();
            batch.set(ticketRef, newTicket);
            batch.update(db.collection('leaveRequests').doc(leaveRequest.id), {
                status: 'ApprovedWithConflict',
                conflictTicketId: ticketRef.id,
                updatedAt: admin.firestore.FieldValue.serverTimestamp()
            });
            
            await batch.commit();
        }
    } catch (error) {
        console.error(`Failed to check conflicts for leave request ${leaveRequest.id}:`, error);
        throw error;
    }
}

async function getAssignmentsForPeriod(
    collectionName: string, 
    leave: LeaveRequest, 
    subField?: string
): Promise<any[]> {
    const assignments: any[] = [];
    
    // Generate document IDs for the leave period
    const docIds = generateDocIdsForPeriod(
        collectionName, 
        leave.startDate.toDate(), 
        leave.endDate.toDate()
    );

    // Fetch all relevant documents
    const docPromises = docIds.map(id => 
        db.collection(collectionName).doc(id).get()
    );
    
    const docs = await Promise.all(docPromises);
    
    for (const doc of docs) {
        if (!doc.exists) continue;
        
        const data = doc.data();
        if (!data) continue;

        if (collectionName === 'monthlySchedules') {
            // Monthly schedule has assignments array
            const monthlyAssignments = data.assignments || [];
            const residentAssignments = monthlyAssignments.filter((a: any) => 
                a.residentId === leave.residentId &&
                a.date.toDate() >= leave.startDate.toDate() &&
                a.date.toDate() <= leave.endDate.toDate()
            );
            assignments.push(...residentAssignments);
        } else if (collectionName === 'weeklySchedules' && subField) {
            // Weekly schedule has nested structure
            const days = data.days || [];
            for (const day of days) {
                const dayDate = day.date.toDate();
                if (dayDate >= leave.startDate.toDate() && 
                    dayDate <= leave.endDate.toDate()) {
                    
                    const dayAssignments = day.assignments?.[subField] || [];
                    const residentAssignments = dayAssignments.filter((a: any) => 
                        a.residentId === leave.residentId
                    );
                    assignments.push(...residentAssignments);
                }
            }
        }
    }
    
    return assignments;
}

function generateDocIdsForPeriod(
    collectionName: string, 
    startDate: Date, 
    endDate: Date
): string[] {
    const ids: string[] = [];
    const current = new Date(startDate);
    
    while (current <= endDate) {
        if (collectionName === 'monthlySchedules') {
            // Format: YYYY-MM
            const year = current.getFullYear();
            const month = (current.getMonth() + 1).toString().padStart(2, '0');
            ids.push(`${year}-${month}`);
            current.setMonth(current.getMonth() + 1);
        } else if (collectionName === 'weeklySchedules') {
            // Format: YYYY-WW
            const year = current.getFullYear();
            const week = getWeekNumber(current);
            ids.push(`${year}-${week}`);
            current.setDate(current.getDate() + 7);
        }
    }
    
    return [...new Set(ids)]; // Remove duplicates
}

function getWeekNumber(date: Date): number {
    const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
    d.setUTCDate(d.getUTCDate() + 4 - (d.getUTCDay() || 7));
    const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
    const weekNo = Math.ceil(((d.valueOf() - yearStart.valueOf()) / 86400000 + 1) / 7);
    return weekNo;
}

function calculatePriority(conflicts: ConflictingAssignment[]): 'High' | 'Medium' | 'Low' {
    // High priority if conflicts with call or multiple conflicts
    if (conflicts.some(c => c.type === 'Call') || conflicts.length > 3) {
        return 'High';
    }
    // Medium priority for OR conflicts
    if (conflicts.some(c => c.type === 'OR')) {
        return 'Medium';
    }
    // Low priority for clinic only
    return 'Low';
}