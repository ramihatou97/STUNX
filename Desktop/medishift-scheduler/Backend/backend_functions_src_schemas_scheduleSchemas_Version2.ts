import { z } from 'zod';

// Resident schema
const ResidentSchema = z.object({
    id: z.string().min(1),
    name: z.string().min(1),
    email: z.string().email(),
    pgyLevel: z.number().min(1).max(7),
    specialty: z.enum(['Neurosurgery', 'Plastics', 'Orthopedics', 'General']),
    onService: z.boolean(),
    isChief: z.boolean(),
    callExempt: z.boolean(),
    team: z.enum(['Red', 'Blue']).optional()
});

// External rotator schema
const ExternalRotatorSchema = z.object({
    id: z.string().min(1),
    name: z.string().min(1),
    email: z.string().email(),
    homeInstitution: z.string(),
    pgyLevel: z.number().min(1).max(7),
    startDate: z.string().datetime(),
    endDate: z.string().datetime()
});

// Configuration schema
const ConfigurationSchema = z.object({
    monthlySchedulerConfig: z.object({
        paroHardCaps: z.array(z.object({
            minDays: z.number(),
            maxDays: z.number(),
            calls: z.number()
        })),
        callRatios: z.record(z.number()),
        maxWeekendsPerRotation: z.number(),
        weekendDefinition: z.array(z.string())
    }),
    yearlySchedulerConfig: z.object({
        mandatoryRotations: z.array(z.any()),
        examLeave: z.array(z.any())
    }),
    holidays: z.array(z.string())
});

// Yearly schedule request schema
export const YearlyScheduleSchema = z.object({
    academicYearId: z.string().regex(/^\d{4}-\d{4}$/),
    residents: z.array(ResidentSchema),
    externalRotators: z.array(ExternalRotatorSchema),
    config: ConfigurationSchema,
    forceRegenerate: z.boolean().optional()
});

// Monthly schedule request schema
export const MonthlyScheduleSchema = z.object({
    month: z.number().min(0).max(11),
    year: z.number().min(2020).max(2030),
    staffingLevel: z.enum(['Normal', 'Shortage']).optional(),
    forceRegenerate: z.boolean().optional(),
    debugMode: z.boolean().optional()
});

// Weekly schedule request schema
export const WeeklyScheduleSchema = z.object({
    weekStartDate: z.string().datetime(),
    residents: z.array(ResidentSchema),
    orSlots: z.array(z.object({
        id: z.string(),
        date: z.string().datetime(),
        room: z.string(),
        startTime: z.string(),
        endTime: z.string()
    })),
    clinicSlots: z.array(z.object({
        id: z.string(),
        date: z.string().datetime(),
        clinicName: z.string(),
        location: z.string()
    })),
    callAssignments: z.array(z.any()).optional(),
    config: ConfigurationSchema
});