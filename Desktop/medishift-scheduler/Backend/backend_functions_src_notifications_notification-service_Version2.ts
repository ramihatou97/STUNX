import * as functions from 'firebase-functions';
import * as admin from 'firebase-admin';
import { 
    LeaveRequest, 
    EPAssignment, 
    ConflictResolutionTicket,
    Notification
} from '../../../shared/types';

const db = admin.firestore();

type NotificationType = 'LeaveRequest' | 'EPA' | 'Conflict' | 'Schedule' | 'System';

interface NotificationData {
    recipientId: string;
    title: string;
    message: string;
    type: NotificationType;
    linkTo?: string;
    priority?: 'Low' | 'Medium' | 'High';
    metadata?: Record<string, any>;
}

/**
 * Central notification creation service
 */
async function createNotification(data: NotificationData): Promise<void> {
    try {
        const notificationRef = db.collection('notifications').doc();
        const newNotification: Notification = {
            id: notificationRef.id,
            recipientId: data.recipientId,
            title: data.title,
            message: data.message,
            type: data.type,
            linkTo: data.linkTo || '',
            isRead: false,
            createdAt: admin.firestore.Timestamp.now(),
            priority: data.priority || 'Low',
            metadata: data.metadata || {}
        };

        await notificationRef.set(newNotification);
        console.log(`📬 Notification created for ${data.recipientId}: ${data.title}`);
        
        // TODO: Integrate with email service (SendGrid)
        // TODO: Integrate with push notifications (FCM)
        
    } catch (error) {
        console.error('Failed to create notification:', error);
        throw error;
    }
}

/**
 * Batch create notifications for multiple recipients
 */
async function createBatchNotifications(
    recipientIds: string[], 
    data: Omit<NotificationData, 'recipientId'>
): Promise<void> {
    const batch = db.batch();
    
    recipientIds.forEach(recipientId => {
        const notificationRef = db.collection('notifications').doc();
        const notification: Notification = {
            id: notificationRef.id,
            recipientId,
            title: data.title,
            message: data.message,
            type: data.type,
            linkTo: data.linkTo || '',
            isRead: false,
            createdAt: admin.firestore.Timestamp.now(),
            priority: data.priority || 'Low',
            metadata: data.metadata || {}
        };
        batch.set(notificationRef, notification);
    });
    
    await batch.commit();
    console.log(`📬 Batch notifications created for ${recipientIds.length} recipients`);
}

/**
 * Send notification when leave request status changes
 */
export const onLeaveRequestStatusChange = functions.firestore
    .document('leaveRequests/{requestId}')
    .onUpdate(async (change, context) => {
        const before = change.before.data() as LeaveRequest;
        const after = change.after.data() as LeaveRequest;

        if (before.status === after.status) {
            return null; // No status change
        }

        let notificationData: NotificationData | null = null;

        switch (after.status) {
            case 'Approved':
                notificationData = {
                    recipientId: after.residentId,
                    title: '✅ Vacation Request Approved',
                    message: `Your ${after.type} leave request from ${
                        after.startDate.toDate().toLocaleDateString()
                    } to ${
                        after.endDate.toDate().toLocaleDateString()
                    } has been approved.`,
                    type: 'LeaveRequest',
                    linkTo: `/vacation/${after.id}`,
                    priority: 'Medium'
                };
                break;
                
            case 'Denied':
                notificationData = {
                    recipientId: after.residentId,
                    title: '❌ Vacation Request Denied',
                    message: `Your ${after.type} leave request was denied. ${
                        after.denialJustification ? `Reason: ${after.denialJustification}` : ''
                    }`,
                    type: 'LeaveRequest',
                    linkTo: `/vacation/${after.id}`,
                    priority: 'High'
                };
                break;
                
            case 'ApprovedWithConflict':
                notificationData = {
                    recipientId: after.residentId,
                    title: '⚠️ Schedule Conflict Detected',
                    message: `Your approved leave has a scheduling conflict that needs resolution.`,
                    type: 'Conflict',
                    linkTo: `/vacation/${after.id}`,
                    priority: 'High'
                };
                break;
        }

        if (notificationData) {
            await createNotification(notificationData);
        }
        
        return null;
    });

/**
 * Send notification when EPA is assigned
 */
export const onEpaAssigned = functions.firestore
    .document('epas/{epaId}')
    .onCreate(async (snap, context) => {
        const epa = snap.data() as EPAssignment;
        
        await createNotification({
            recipientId: epa.residentId,
            title: '📚 New EPA Assigned',
            message: `An EPA for "${epa.epaName || epa.epaId}" has been assigned to you based on a recent clinical case.`,
            type: 'EPA',
            linkTo: `/epas/${epa.id}`,
            priority: 'Medium',
            metadata: {
                epaId: epa.epaId,
                caseId: epa.caseId,
                dueDate: epa.dueDate.toDate().toISOString()
            }
        });
        
        return null;
    });

/**
 * Send notification to admins when conflict is detected
 */
export const onConflictDetected = functions.firestore
    .document('conflictTickets/{ticketId}')
    .onCreate(async (snap, context) => {
        const ticket = snap.data() as ConflictResolutionTicket;
        
        // Fetch admin user IDs
        const adminIds = await getAdminUserIds();
        
        if (adminIds.length === 0) {
            console.warn('No admin users found to notify about conflict');
            return null;
        }
        
        await createBatchNotifications(adminIds, {
            title: '🚨 Action Required: Schedule Conflict',
            message: `${ticket.residentName}'s approved vacation conflicts with a published schedule. Please resolve immediately.`,
            type: 'Conflict',
            linkTo: `/admin/conflicts/${ticket.id}`,
            priority: 'High',
            metadata: {
                ticketId: ticket.id,
                residentId: ticket.residentId,
                residentName: ticket.residentName,
                conflictCount: ticket.conflictingAssignments.length
            }
        });
        
        return null;
    });

/**
 * Clean up old read notifications
 */
export const cleanupOldNotifications = functions.pubsub
    .schedule('every sunday 02:00')
    .timeZone('America/New_York')
    .onRun(async (context) => {
        console.log('🧹 Cleaning up old notifications...');
        
        const thirtyDaysAgo = new Date();
        thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
        
        const oldNotifications = await db.collection('notifications')
            .where('isRead', '==', true)
            .where('createdAt', '<', admin.firestore.Timestamp.fromDate(thirtyDaysAgo))
            .limit(500) // Process in batches
            .get();
        
        if (oldNotifications.empty) {
            console.log('No old notifications to clean up');
            return null;
        }
        
        const batch = db.batch();
        oldNotifications.docs.forEach(doc => {
            batch.delete(doc.ref);
        });
        
        await batch.commit();
        console.log(`✅ Deleted ${oldNotifications.size} old notifications`);
        
        return null;
    });

/**
 * Helper function to get admin user IDs
 */
async function getAdminUserIds(): Promise<string[]> {
    try {
        // Option 1: Get from a dedicated admins collection
        const adminsSnapshot = await db.collection('admins').get();
        if (!adminsSnapshot.empty) {
            return adminsSnapshot.docs.map(doc => doc.id);
        }
        
        // Option 2: Get users with admin custom claims
        const listUsersResult = await admin.auth().listUsers(1000);
        const adminUsers = listUsersResult.users.filter(user => 
            user.customClaims?.admin === true
        );
        
        return adminUsers.map(user => user.uid);
        
    } catch (error) {
        console.error('Failed to fetch admin user IDs:', error);
        return [];
    }
}