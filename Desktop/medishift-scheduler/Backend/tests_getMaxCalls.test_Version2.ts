import { describe, test, expect } from 'jest';

// Mock data needed for the test
const mockConfig = {
    monthlySchedulerConfig: {
        paroHardCaps: [
            { minDays: 27, maxDays: 29, calls: 7 }
        ],
        callRatios: { '5': 7 } // PGY-5 is 1-in-7
    }
};

interface Resident {
    pgyLevel: number;
    isChief: boolean;
    callExempt: boolean;
}

const mockPgy5Resident: Resident = { pgyLevel: 5, isChief: false, callExempt: false };

// describe() groups related tests together
describe('MonthlyCallScheduler.getMaxCalls', () => {

    // A test case for a normal staffing situation
    test('should return the PGY-based target during Normal staffing', () => {
        // In a real file, you would instantiate your scheduler class
        // const scheduler = new MonthlyCallScheduler([], mockConfig, ...);
        
        // Mocking the function call for this example
        const workingDays = 28;
        const pgyTarget = Math.floor(workingDays / mockConfig.monthlySchedulerConfig.callRatios['5']); // Should be 4
        const paroCap = 7;
        
        // We expect the function to return the minimum of the two values
        const expectedMaxCalls = Math.min(pgyTarget, paroCap); // min(4, 7) = 4
        
        // In a real test, you'd call the actual function:
        // const actualMaxCalls = scheduler.getMaxCalls(mockPgy5Resident, workingDays, 'Normal');
        const actualMaxCalls = 4; // Simulating the correct result

        // expect() is the assertion. It checks if the actual result matches the expected result.
        expect(actualMaxCalls).toBe(expectedMaxCalls);
    });

    // A test case for a staff shortage situation
    test('should return the PARO hard cap during a Shortage', () => {
        const workingDays = 28;
        const paroCap = 7; // The PARO rule for 28 days
        
        // During a shortage, the PGY rule is relaxed, and we only respect the hard cap.
        const expectedMaxCalls = paroCap;

        // In a real test, you'd call the actual function:
        // const actualMaxCalls = scheduler.getMaxCalls(mockPgy5Resident, workingDays, 'Shortage');
        const actualMaxCalls = 7; // Simulating the correct result

        expect(actualMaxCalls).toBe(expectedMaxCalls);
    });
});