import { z } from 'zod';
import * as functions from 'firebase-functions';

export function validateRequest<T>(schema: z.ZodSchema<T>) {
    return (data: any): T => {
        try {
            return schema.parse(data);
        } catch (error) {
            if (error instanceof z.ZodError) {
                const details = error.errors.map(e => ({
                    path: e.path.join('.'),
                    message: e.message
                }));
                
                throw new functions.https.HttpsError(
                    'invalid-argument',
                    `Validation failed: ${details.map(d => 
                        `${d.path}: ${d.message}`
                    ).join(', ')}`,
                    { details }
                );
            }
            throw error;
        }
    };
}

// Async validator for complex validations
export async function validateRequestAsync<T>(
    schema: z.ZodSchema<T>,
    customValidations?: (data: T) => Promise<void>
) {
    return async (data: any): Promise<T> => {
        const validated = validateRequest(schema)(data);
        
        if (customValidations) {
            await customValidations(validated);
        }
        
        return validated;
    };
}