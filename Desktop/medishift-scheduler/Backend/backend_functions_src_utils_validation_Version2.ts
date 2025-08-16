import { z } from 'zod';
import { Timestamp } from 'firebase-admin/firestore';

// Leave Request validation schema
export const LeaveRequestSchema = z.object({
    residentId: z.string().min(1),
    type: z.enum(['Personal', 'Professional', 'LieuDay', 'Compassionate']),
    startDate: z.instanceof(Timestamp),
    endDate: z.instanceof(Timestamp),
    reason: z.string().optional(),
}).refine(data => data.endDate.toDate() >= data.startDate.toDate(), {
    message: "End date must be after start date"
});

// Schedule generation request validation
export const ScheduleRequestSchema = z.object({
    period: z.enum(['yearly', 'monthly', 'weekly']),
    year: z.number().min(2020).max(2030),
    month: z.number().min(0).max(11).optional(),
    week: z.number().min(1).max(53).optional(),
    staffingLevel: z.enum(['Normal', 'Shortage']).default('Normal')
});

// Validate function wrapper
export function validateInput<T>(schema: z.ZodSchema<T>) {
    return (target: any, propertyKey: string, descriptor: PropertyDescriptor) => {
        const originalMethod = descriptor.value;
        
        descriptor.value = async function(data: any, context: any) {
            try {
                const validated = schema.parse(data);
                return await originalMethod.call(this, validated, context);
            } catch (error) {
                if (error instanceof z.ZodError) {
                    throw new functions.https.HttpsError(
                        'invalid-argument',
                        `Validation error: ${error.errors.map(e => e.message).join(', ')}`
                    );
                }
                throw error;
            }
        };
        
        return descriptor;
    };
}