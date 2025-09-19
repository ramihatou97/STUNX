/**
 * Monthly Call Scheduler Test Suite
 * Tests all critical functionality including getMaxCalls logic
 * Author: MediShift Team
 * Date: January 2025
 */

import { MonthlyCallScheduler } from '../scheduling/monthly-scheduler';
import { 
    Resident, 
    AppConfiguration, 
    AcademicYear, 
    LeaveRequest,
    CallAssignment,
    RotationBlock 
} from '../../../shared/types';
import { Timestamp } from 'firebase-admin/firestore';

// ===================================================================
// MOCK DATA SETUP
// ===================================================================

const createMockResident = (overrides: Partial<Resident> = {}): Resident => ({
    id: 'res-001',
    name: 'Dr. Smith',
    email: 'smith@hospital.com',
    pgyLevel: 3,
    specialty: 'Neurosurgery',
    isChief: false,
    callExempt: false,
    onService: true,
    ...overrides
});

const createMockConfig = (): AppConfiguration => ({
    monthlySchedulerConfig: {
        paroHardCaps: [
            { minDays: 20, maxDays: 22, calls: 5 },
            { minDays: 23, maxDays: 25, calls: 6 },
            { minDays: 26, maxDays: 28, calls: 7 },
            { minDays: 29, maxDays: 31, calls: 8 }
        ],
        callRatios: {
            1: 4,  // PGY-1: 1-in-4
            2: 5,  // PGY-2: 1-in-5
            3: 6,  // PGY-3: 1-in-6
            4: 7,  // PGY-4: 1-in-7
            5: 7   // PGY-5: 1-in-7
        },
        maxWeekendsPerRotation: 2,
        weekendDefinition: ['Friday', 'Saturday', 'Sunday']
    },
    holidays: [
        '2025-08-04',  // Civic Holiday
        '2025-09-01',  // Labor Day
    ],
    leavePolicy: {
        minNoticeDays: 30,
        maxConsecutiveDays: 14,
        annualLimit: 21
    }
} as AppConfiguration);

const createMockAcademicYear = (): AcademicYear => {
    const blocks: RotationBlock[] = [];
    
    // Create blocks for the academic year
    for (let i = 0; i < 13; i++) {
        const startDate = new Date(2025, 6 + Math.floor(i * 28 / 30), 1 + (i * 28) % 30);
        const endDate = new Date(startDate);
        endDate.setDate(endDate.getDate() + 27);
        
        blocks.push({
            blockNumber: i + 1,
            startDate: Timestamp.fromDate(startDate),
            endDate: Timestamp.fromDate(endDate),
            assignments: []
        });
    }
    
    return {
        id: '2025-2026',
        blocks,
        metadata: {
            generatedAt: Timestamp.now(),
            totalResidents: 15,
            totalExternalRotators: 2,
            version: '1.0.0'
        }
    };
};

// ===================================================================
// TEST SUITE
// ===================================================================

describe('MonthlyCallScheduler', () => {
    let scheduler: MonthlyCallScheduler;
    let mockResidents: Resident[];
    let mockConfig: AppConfiguration;
    let mockAcademicYear: AcademicYear;
    let mockLeaveRequests: LeaveRequest[];

    beforeEach(() => {
        // Setup mock residents with different PGY levels
        mockResidents = [
            createMockResident({ id: 'res-001', name: 'Dr. Chen', pgyLevel: 1 }),
            createMockResident({ id: 'res-002', name: 'Dr. Smith', pgyLevel: 2 }),
            createMockResident({ id: 'res-003', name: 'Dr. Johnson', pgyLevel: 3 }),
            createMockResident({ id: 'res-004', name: 'Dr. Williams', pgyLevel: 4 }),
            createMockResident({ id: 'res-005', name: 'Dr. Brown', pgyLevel: 5 }),
            createMockResident({ 
                id: 'res-006', 
                name: 'Dr. Chief', 
                pgyLevel: 5, 
                isChief: true, 
                callExempt: true 
            }),
        ];

        mockConfig = createMockConfig();
        mockAcademicYear = createMockAcademicYear();
        mockLeaveRequests = [];

        // Add residents to rotation blocks
        mockAcademicYear.blocks.forEach(block => {
            mockResidents.forEach(resident => {
                block.assignments.push({
                    residentId: resident.id,
                    rotationName: 'Neurosurgery - Core',
                    rotationType: 'CORE_NSX',
                    team: resident.pgyLevel <= 3 ? 'Red' : 'Blue'
                });
            });
        });
    });

    // ===================================================================
    // getMaxCalls LOGIC TESTS
    // ===================================================================

    describe('getMaxCalls Logic', () => {
        test('should return 0 for call-exempt chief', () => {
            const chiefResident = mockResidents.find(r => r.isChief && r.callExempt)!;
            
            scheduler = new MonthlyCallScheduler(
                [chiefResident],
                mockConfig,
                mockAcademicYear,
                [],
                7, // August
                2025
            );

            const assignments = scheduler.generateSchedule('Normal');
            const chiefCalls = assignments.filter(a => 
                a.residentId === chiefResident.id && a.type !== 'PostCall'
            );

            expect(chiefCalls.length).toBe(0);
            console.log(`✅ Chief ${chiefResident.name} correctly assigned 0 calls`);
        });

        test('should apply PGY-based targets in Normal mode', () => {
            const pgy1Resident = mockResidents.find(r => r.pgyLevel === 1)!;
            const pgy5Resident = mockResidents.find(r => r.pgyLevel === 5 && !r.isChief)!;
            
            scheduler = new MonthlyCallScheduler(
                [pgy1Resident, pgy5Resident],
                mockConfig,
                mockAcademicYear,
                [],
                7, // August (28 working days)
                2025
            );

            const assignments = scheduler.generateSchedule('Normal');
            
            const pgy1Calls = assignments.filter(a => 
                a.residentId === pgy1Resident.id && a.type !== 'PostCall'
            );
            const pgy5Calls = assignments.filter(a => 
                a.residentId === pgy5Resident.id && a.type !== 'PostCall'
            );

            // PGY-1: 28/4 = 7, PARO cap = 7, min(7,7) = 7
            // PGY-5: 28/7 = 4, PARO cap = 7, min(4,7) = 4
            
            expect(pgy1Calls.length).toBeLessThanOrEqual(7);
            expect(pgy5Calls.length).toBeLessThanOrEqual(4);
            
            console.log(`✅ PGY-1: ${pgy1Calls.length} calls (max 7)`);
            console.log(`✅ PGY-5: ${pgy5Calls.length} calls (max 4)`);
        });

        test('should relax to PARO caps in Shortage mode', () => {
            const pgy5Resident = mockResidents.find(r => r.pgyLevel === 5 && !r.isChief)!;
            
            scheduler = new MonthlyCallScheduler(
                [pgy5Resident],
                mockConfig,
                mockAcademicYear,
                [],
                7, // August
                2025
            );

            const shortageAssignments = scheduler.generateSchedule('Shortage');
            
            const pgy5ShortCalls = shortageAssignments.filter(a => 
                a.residentId === pgy5Resident.id && a.type !== 'PostCall'
            );

            // In shortage mode: should use PARO cap (7) instead of PGY target (4)
            expect(pgy5ShortCalls.length).toBeLessThanOrEqual(7);
            expect(pgy5ShortCalls.length).toBeGreaterThan(4); // Should be more than normal mode
            
            console.log(`✅ Shortage mode: ${pgy5ShortCalls.length} calls (PARO cap: 7)`);
        });

        test('should respect PARO hard caps based on working days', () => {
            // Test with different month lengths
            const testCases = [
                { month: 1, workingDays: 20, expectedMax: 5 }, // February (short)
                { month: 7, workingDays: 28, expectedMax: 7 }, // August (standard)
                { month: 9, workingDays: 30, expectedMax: 8 }, // October (long)
            ];

            testCases.forEach(({ month, workingDays, expectedMax }) => {
                scheduler = new MonthlyCallScheduler(
                    [mockResidents[0]],
                    mockConfig,
                    mockAcademicYear,
                    [],
                    month,
                    2025
                );

                const assignments = scheduler.generateSchedule('Shortage');
                const calls = assignments.filter(a => 
                    a.residentId === mockResidents[0].id && a.type !== 'PostCall'
                );

                expect(calls.length).toBeLessThanOrEqual(expectedMax);
                console.log(`✅ Month ${month}: ${calls.length} calls (max ${expectedMax})`);
            });
        });
    });

    // ===================================================================
    // WEEKEND LIMIT TESTS
    // ===================================================================

    describe('Weekend Limits', () => {
        test('should not exceed maximum weekends per rotation', () => {
            scheduler = new MonthlyCallScheduler(
                mockResidents.filter(r => !r.callExempt),
                mockConfig,
                mockAcademicYear,
                [],
                7, // August
                2025
            );

            const assignments = scheduler.generateSchedule('Normal');
            
            mockResidents.forEach(resident => {
                const weekendCalls = assignments.filter(a => 
                    a.residentId === resident.id && 
                    a.type === 'Weekend'
                );
                
                expect(weekendCalls.length).toBeLessThanOrEqual(2);
                console.log(`✅ ${resident.name}: ${weekendCalls.length} weekend calls (max 2)`);
            });
        });
    });

    // ===================================================================
    // LEAVE CONFLICT TESTS
    // ===================================================================

    describe('Leave Conflicts', () => {
        test('should not assign calls during approved leave', () => {
            const leaveResident = mockResidents[0];
            
            mockLeaveRequests = [{
                id: 'leave-001',
                residentId: leaveResident.id,
                residentName: leaveResident.name,
                type: 'Personal',
                status: 'Approved',
                startDate: Timestamp.fromDate(new Date(2025, 7, 10)), // Aug 10
                endDate: Timestamp.fromDate(new Date(2025, 7, 15)),   // Aug 15
                createdAt: Timestamp.now()
            }];

            scheduler = new MonthlyCallScheduler(
                mockResidents,
                mockConfig,
                mockAcademicYear,
                mockLeaveRequests,
                7, // August
                2025
            );

            const assignments = scheduler.generateSchedule('Normal');
            
            // Check no calls during leave period
            const leavePeriodCalls = assignments.filter(a => {
                const callDate = a.date.toDate();
                return a.residentId === leaveResident.id &&
                       callDate >= new Date(2025, 7, 10) &&
                       callDate <= new Date(2025, 7, 15);
            });

            expect(leavePeriodCalls.length).toBe(0);
            console.log(`✅ No calls assigned during ${leaveResident.name}'s leave`);
        });
    });

    // ===================================================================
    // POST-CALL TESTS
    // ===================================================================

    describe('Post-Call Handling', () => {
        test('should create post-call assignments after night/weekend calls', () => {
            scheduler = new MonthlyCallScheduler(
                mockResidents.slice(0, 2),
                mockConfig,
                mockAcademicYear,
                [],
                7, // August
                2025
            );

            const assignments = scheduler.generateSchedule('Normal');
            
            const nightCalls = assignments.filter(a => a.type === 'Night');
            const postCalls = assignments.filter(a => a.type === 'PostCall');
            
            // Each night call should have a corresponding post-call
            expect(postCalls.length).toBeGreaterThan(0);
            console.log(`✅ Created ${postCalls.length} post-call assignments for ${nightCalls.length} night calls`);
        });

        test('should prevent consecutive calls (respecting post-call)', () => {
            scheduler = new MonthlyCallScheduler(
                mockResidents.slice(0, 1), // Single resident
                mockConfig,
                mockAcademicYear,
                [],
                7, // August
                2025
            );

            const assignments = scheduler.generateSchedule('Normal');
            const resident = mockResidents[0];
            
            // Check no back-to-back calls
            const residentCalls = assignments
                .filter(a => a.residentId === resident.id && a.type !== 'PostCall')
                .sort((a, b) => a.date.toDate().getTime() - b.date.toDate().getTime());
            
            let hasConsecutiveCalls = false;
            for (let i = 1; i < residentCalls.length; i++) {
                const daysBetween = Math.floor(
                    (residentCalls[i].date.toDate().getTime() - 
                     residentCalls[i-1].date.toDate().getTime()) / 
                    (1000 * 60 * 60 * 24)
                );
                if (daysBetween < 2) {
                    hasConsecutiveCalls = true;
                    break;
                }
            }
            
            expect(hasConsecutiveCalls).toBe(false);
            console.log(`✅ No consecutive calls for ${resident.name}`);
        });
    });

    // ===================================================================
    // FAIRNESS DISTRIBUTION TESTS
    // ===================================================================

    describe('Fairness Distribution', () => {
        test('should distribute calls fairly among residents', () => {
            const eligibleResidents = mockResidents.filter(r => !r.callExempt);
            
            scheduler = new MonthlyCallScheduler(
                eligibleResidents,
                mockConfig,
                mockAcademicYear,
                [],
                7, // August
                2025
            );

            const assignments = scheduler.generateSchedule('Normal');
            
            // Calculate distribution
            const callCounts = new Map<string, number>();
            eligibleResidents.forEach(r => callCounts.set(r.id, 0));
            
            assignments.forEach(a => {
                if (a.type !== 'PostCall') {
                    const count = callCounts.get(a.residentId) || 0;
                    callCounts.set(a.residentId, count + 1);
                }
            });
            
            // Calculate standard deviation
            const counts = Array.from(callCounts.values());
            const mean = counts.reduce((a, b) => a + b, 0) / counts.length;
            const variance = counts.reduce((sum, count) => 
                sum + Math.pow(count - mean, 2), 0
            ) / counts.length;
            const stdDev = Math.sqrt(variance);
            
            // Should have low standard deviation (fair distribution)
            expect(stdDev).toBeLessThan(2);
            
            console.log(`✅ Fair distribution achieved:`);
            eligibleResidents.forEach(r => {
                console.log(`   ${r.name}: ${callCounts.get(r.id)} calls`);
            });
            console.log(`   Standard deviation: ${stdDev.toFixed(2)}`);
        });
    });

    // ===================================================================
    // HOLIDAY COVERAGE TESTS
    // ===================================================================

    describe('Holiday Coverage', () => {
        test('should assign appropriate coverage for holidays', () => {
            scheduler = new MonthlyCallScheduler(
                mockResidents,
                mockConfig,
                mockAcademicYear,
                [],
                7, // August (has Civic Holiday on Aug 4)
                2025
            );

            const assignments = scheduler.generateSchedule('Normal');
            
            const holidayAssignments = assignments.filter(a => 
                a.isHoliday && a.type === 'Holiday'
            );
            
            expect(holidayAssignments.length).toBeGreaterThan(0);
            console.log(`✅ Holiday coverage: ${holidayAssignments.length} assignments`);
        });
    });

    // ===================================================================
    // PARO COMPLIANCE TESTS
    // ===================================================================

    describe('PARO 24-Hour Rule Compliance', () => {
        test('should respect 1-in-4 days averaged rule', () => {
            scheduler = new MonthlyCallScheduler(
                mockResidents.slice(0, 3),
                mockConfig,
                mockAcademicYear,
                [],
                7, // August
                2025
            );

            const assignments = scheduler.generateSchedule('Normal');
            
            // Check each resident's call frequency
            mockResidents.slice(0, 3).forEach(resident => {
                const residentCalls = assignments.filter(a => 
                    a.residentId === resident.id && a.type !== 'PostCall'
                );
                
                // Over 28 days, max should be 7 (28/4)
                expect(residentCalls.length).toBeLessThanOrEqual(7);
                
                const ratio = residentCalls.length > 0 ? 28 / residentCalls.length : 999;
                console.log(`✅ ${resident.name}: 1-in-${ratio.toFixed(1)} call ratio`);
            });
        });
    });

    // ===================================================================
    // EDGE CASE TESTS
    // ===================================================================

    describe('Edge Cases', () => {
        test('should handle single resident scenario', () => {
            scheduler = new MonthlyCallScheduler(
                [mockResidents[0]],
                mockConfig,
                mockAcademicYear,
                [],
                7,
                2025
            );

            const assignments = scheduler.generateSchedule('Shortage');
            
            expect(assignments.length).toBeGreaterThan(0);
            expect(assignments.length).toBeLessThanOrEqual(15); // Max with post-calls
            console.log(`✅ Single resident handled: ${assignments.length} assignments`);
        });

        test('should handle empty resident list gracefully', () => {
            scheduler = new MonthlyCallScheduler(
                [],
                mockConfig,
                mockAcademicYear,
                [],
                7,
                2025
            );

            const assignments = scheduler.generateSchedule('Normal');
            
            expect(assignments).toEqual([]);
            console.log(`✅ Empty resident list handled gracefully`);
        });

        test('should handle all residents on leave', () => {
            const allOnLeave = mockResidents.map(r => ({
                id: `leave-${r.id}`,
                residentId: r.id,
                residentName: r.name,
                type: 'Personal' as const,
                status: 'Approved' as const,
                startDate: Timestamp.fromDate(new Date(2025, 7, 1)),
                endDate: Timestamp.fromDate(new Date(2025, 7, 31)),
                createdAt: Timestamp.now()
            }));

            scheduler = new MonthlyCallScheduler(
                mockResidents,
                mockConfig,
                mockAcademicYear,
                allOnLeave,
                7,
                2025
            );

            const assignments = scheduler.generateSchedule('Normal');
            
            expect(assignments.length).toBe(0);
            console.log(`✅ All residents on leave handled: no assignments created`);
        });
    });

    // ===================================================================
    // PERFORMANCE TESTS
    // ===================================================================

    describe('Performance', () => {
        test('should complete scheduling within reasonable time', () => {
            // Create larger resident pool
            const manyResidents = Array.from({ length: 30 }, (_, i) => 
                createMockResident({
                    id: `res-perf-${i}`,
                    name: `Dr. Test${i}`,
                    pgyLevel: (i % 5) + 1
                })
            );

            // Add to academic year
            mockAcademicYear.blocks.forEach(block => {
                manyResidents.forEach(resident => {
                    block.assignments.push({
                        residentId: resident.id,
                        rotationName: 'Neurosurgery - Core',
                        rotationType: 'CORE_NSX'
                    });
                });
            });

            const startTime = Date.now();
            
            scheduler = new MonthlyCallScheduler(
                manyResidents,
                mockConfig,
                mockAcademicYear,
                [],
                7,
                2025
            );

            const assignments = scheduler.generateSchedule('Normal');
            
            const endTime = Date.now();
            const duration = endTime - startTime;
            
            expect(duration).toBeLessThan(5000); // Should complete within 5 seconds
            console.log(`✅ Scheduled ${manyResidents.length} residents in ${duration}ms`);
            console.log(`   Generated ${assignments.length} assignments`);
        });
    });

    // ===================================================================
    // INTEGRATION TESTS
    // ===================================================================

    describe('Integration Tests', () => {
        test('should generate valid complete monthly schedule', () => {
            scheduler = new MonthlyCallScheduler(
                mockResidents,
                mockConfig,
                mockAcademicYear,
                [],
                7, // August
                2025,
                [],
                true // Enable debug mode
            );

            const assignments = scheduler.generateSchedule('Normal');
            
            // Validate schedule completeness
            expect(assignments).toBeDefined();
            expect(assignments.length).toBeGreaterThan(0);
            
            // Check all assignments have required fields
            assignments.forEach(a => {
                expect(a.id).toBeDefined();
                expect(a.residentId).toBeDefined();
                expect(a.residentName).toBeDefined();
                expect(a.date).toBeDefined();
                expect(a.type).toBeDefined();
                expect(a.points).toBeDefined();
            });
            
            // Verify no undefined residents
            const validResidentIds = new Set(mockResidents.map(r => r.id));
            assignments.forEach(a => {
                expect(validResidentIds.has(a.residentId)).toBe(true);
            });
            
            console.log(`✅ Generated valid complete schedule with ${assignments.length} assignments`);
        });

        test('should handle Normal to Shortage mode transition', () => {
            scheduler = new MonthlyCallScheduler(
                mockResidents.filter(r => !r.callExempt),
                mockConfig,
                mockAcademicYear,
                [],
                7,
                2025
            );

            const normalAssignments = scheduler.generateSchedule('Normal');
            const shortageAssignments = scheduler.generateSchedule('Shortage');
            
            // Shortage mode should allow more calls
            const normalTotal = normalAssignments.filter(a => a.type !== 'PostCall').length;
            const shortageTotal = shortageAssignments.filter(a => a.type !== 'PostCall').length;
            
            expect(shortageTotal).toBeGreaterThanOrEqual(normalTotal);
            
            console.log(`✅ Mode transition handled:`);
            console.log(`   Normal mode: ${normalTotal} calls`);
            console.log(`   Shortage mode: ${shortageTotal} calls`);
        });
    });
});

// ===================================================================
// TEST UTILITIES
// ===================================================================

describe('Test Utilities', () => {
    test('should verify test data integrity', () => {
        const config = createMockConfig();
        const academicYear = createMockAcademicYear();
        
        // Verify config
        expect(config.monthlySchedulerConfig).toBeDefined();
        expect(config.monthlySchedulerConfig.paroHardCaps).toHaveLength(4);
        expect(config.monthlySchedulerConfig.callRatios).toBeDefined();
        
        // Verify academic year
        expect(academicYear.blocks).toHaveLength(13);
        academicYear.blocks.forEach((block, i) => {
            expect(block.blockNumber).toBe(i + 1);
            expect(block.startDate).toBeDefined();
            expect(block.endDate).toBeDefined();
        });
        
        console.log('✅ Test data integrity verified');
    });
});