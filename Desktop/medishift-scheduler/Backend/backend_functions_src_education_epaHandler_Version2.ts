import * as functions from 'firebase-functions';
import * as admin from 'firebase-admin';
import { 
    ORCase, 
    EPADefinition, 
    EPAssignment, 
    Resident 
} from '../../../shared/types';

const db = admin.firestore();

/**
 * EPA Assignment Engine
 * Triggered when an OR case is finalized with assigned residents
 */
export const onORCaseFinalized = functions.firestore
    .document('orCases/{caseId}')
    .onCreate(async (snap, context) => {
        const caseData = snap.data() as ORCase;
        const caseId = context.params.caseId;
        
        const { 
            assignedResidentIds, 
            procedureName, 
            primarySurgeonId,
            date,
            duration
        } = caseData;

        if (!assignedResidentIds || assignedResidentIds.length === 0) {
            console.log(`No residents assigned to OR Case ${caseId}. Exiting.`);
            return null;
        }

        console.log(`📚 Matching EPAs for OR case: ${caseId} (${procedureName})`);

        try {
            // Fetch all EPA definitions
            const epaDefsSnapshot = await db.collection('epaDefinitions').get();
            const allEpaDefs = epaDefsSnapshot.docs.map(doc => ({
                id: doc.id,
                ...doc.data()
            } as EPADefinition));

            // Process each assigned resident
            const batchPromises = assignedResidentIds.map(residentId => 
                processResidentEPAs(
                    residentId, 
                    caseId, 
                    procedureName, 
                    primarySurgeonId, 
                    allEpaDefs
                )
            );

            await Promise.all(batchPromises);
            
            console.log(`✅ EPA matching complete for case ${caseId}`);
            return null;
            
        } catch (error) {
            console.error(`❌ Failed to process EPAs for case ${caseId}:`, error);
            throw error;
        }
    });

async function processResidentEPAs(
    residentId: string,
    caseId: string,
    procedureName: string,
    primarySurgeonId: string,
    allEpaDefs: EPADefinition[]
): Promise<void> {
    try {
        // Find matching EPAs based on procedure keywords
        const matchedEpas = allEpaDefs.filter(def => 
            def.procedureKeywords.some(keyword => 
                procedureName.toLowerCase().includes(keyword.toLowerCase())
            )
        );

        if (matchedEpas.length === 0) {
            console.log(`No matching EPAs found for resident ${residentId} in case ${caseId}`);
            return;
        }

        const batch = db.batch();
        let assignmentCount = 0;

        for (const epa of matchedEpas) {
            // Check if resident needs this EPA
            const needsEpa = await residentNeedsEpa(
                residentId, 
                epa.id, 
                epa.minObservations
            );
            
            if (needsEpa) {
                // Create EPA assignment
                const epaAssignmentRef = db.collection('epas').doc();
                const newAssignment: EPAssignment = {
                    id: epaAssignmentRef.id,
                    residentId,
                    epaId: epa.id,
                    epaName: epa.name,
                    caseId,
                    assessorId: primarySurgeonId,
                    status: 'Assigned',
                    dueDate: admin.firestore.Timestamp.fromMillis(
                        Date.now() + (72 * 60 * 60 * 1000) // 72 hours from now
                    ),
                    assignedAt: admin.firestore.Timestamp.now(),
                    procedureName,
                    competencyLevel: epa.competencyLevel,
                    pgyLevel: await getResidentPGYLevel(residentId)
                };
                
                batch.set(epaAssignmentRef, newAssignment);
                assignmentCount++;
                
                console.log(`📋 Assigned EPA ${epa.id} to resident ${residentId} for case ${caseId}`);
            }
        }

        if (assignmentCount > 0) {
            await batch.commit();
            console.log(`✅ Created ${assignmentCount} EPA assignments for resident ${residentId}`);
        }
        
    } catch (error) {
        console.error(`Failed to process EPAs for resident ${residentId}:`, error);
        throw error;
    }
}

async function residentNeedsEpa(
    residentId: string, 
    epaId: string, 
    requiredCount: number
): Promise<boolean> {
    try {
        const existingAssignments = await db.collection('epas')
            .where('residentId', '==', residentId)
            .where('epaId', '==', epaId)
            .where('status', '==', 'Completed')
            .get();
        
        return existingAssignments.size < requiredCount;
    } catch (error) {
        console.error(`Error checking EPA needs for resident ${residentId}:`, error);
        return false;
    }
}

async function getResidentPGYLevel(residentId: string): Promise<number> {
    try {
        const residentDoc = await db.collection('residents').doc(residentId).get();
        if (!residentDoc.exists) {
            console.warn(`Resident ${residentId} not found`);
            return 1; // Default to PGY-1
        }
        const data = residentDoc.data() as Resident;
        return data.pgyLevel || 1;
    } catch (error) {
        console.error(`Error fetching resident PGY level:`, error);
        return 1;
    }
}

/**
 * Remind residents about pending EPAs
 */
export const sendEPAReminders = functions.pubsub
    .schedule('every day 09:00')
    .timeZone('America/New_York')
    .onRun(async (context) => {
        console.log('📧 Sending EPA reminders...');
        
        try {
            // Find EPAs due within 24 hours
            const tomorrow = new Date();
            tomorrow.setDate(tomorrow.getDate() + 1);
            
            const pendingEPAs = await db.collection('epas')
                .where('status', '==', 'Assigned')
                .where('dueDate', '<=', admin.firestore.Timestamp.fromDate(tomorrow))
                .get();
            
            if (pendingEPAs.empty) {
                console.log('No pending EPAs requiring reminders');
                return null;
            }
            
            const reminderPromises = pendingEPAs.docs.map(doc => {
                const epa = doc.data() as EPAssignment;
                return createEPAReminder(epa);
            });
            
            await Promise.all(reminderPromises);
            
            console.log(`✅ Sent ${pendingEPAs.size} EPA reminders`);
            return null;
            
        } catch (error) {
            console.error('Failed to send EPA reminders:', error);
            throw error;
        }
    });

async function createEPAReminder(epa: EPAssignment): Promise<void> {
    // This would integrate with your notification system
    console.log(`Reminder sent for EPA ${epa.id} to resident ${epa.residentId}`);
}