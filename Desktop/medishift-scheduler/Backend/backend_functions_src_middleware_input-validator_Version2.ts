import { z } from 'zod';
import * as functions from 'firebase-functions';
import * as sanitizeHtml from 'sanitize-html';

// Base schemas for common types
export const ResidentIdSchema = z.string()
    .min(1, 'Resident ID is required')
    .max(50, 'Resident ID too long')
    .regex(/^res-[a-zA-Z0-9]{20}$/, 'Invalid resident ID format');

export const DateSchema = z.string()
    .refine((val) => !isNaN(Date.parse(val)), 'Invalid date format');

export const TimestampSchema = z.object({
    _seconds: z.number(),
    _nanoseconds: z.number()
});

export const MonthSchema = z.number()
    .int('Month must be an integer')
    .min(0, 'Month must be between 0-11')
    .max(11, 'Month must be between 0-11');

export const YearSchema = z.number()
    .int('Year must be an integer')
    .min(2020, 'Year must be 2020 or later')
    .max(2030, 'Year must be 2030 or earlier');

export const PGYLevelSchema = z.number()
    .int('PGY level must be an integer')
    .min(1, 'PGY level must be between 1-5')
    .max(5, 'PGY level must be between 1-5');

// Complex schemas
export const LeaveRequestSchema = z.object({
    residentId: ResidentIdSchema,
    residentName: z.string().min(1).max(100),
    startDate: z.union([DateSchema, TimestampSchema]),
    endDate: z.union([DateSchema, TimestampSchema]),
    type: z.enum(['Vacation', 'Sick', 'Conference', 'Personal', 'Other']),
    reason: z.string().max(500).optional(),
    coverageNeeded: z.boolean(),
    status: z.enum([
        'Pending Analysis',
        'Pending Approval', 
        'Pending Review',
        'Approved',
        'Denied',
        'Cancelled',
        'Analysis Failed'
    ]).optional()
});

export const CallAssignmentSchema = z.object({
    residentId: ResidentIdSchema,
    residentName: z.string().min(1).max(100),
    date: z.union([DateSchema, TimestampSchema]),
    type: z.enum(['Night', 'Weekend', 'Holiday', 'PostCall']),
    points: z.number().min(0).max(10),
    status: z.enum(['Scheduled', 'Completed', 'Missed', 'Swapped', 'Cancelled'])
});

export const GenerateScheduleRequestSchema = z.object({
    month: MonthSchema,
    year: YearSchema,
    staffingLevel: z.enum(['Normal', 'Shortage']).optional(),
    forceRegenerate: z.boolean().optional()
});

// Sanitization functions
export function sanitizeString(input: string): string {
    // Remove any HTML tags and script content
    return sanitizeHtml(input, {
        allowedTags: [],
        allowedAttributes: {},
        disallowedTagsMode: 'discard'
    }).trim();
}

export function sanitizeObject(obj: any): any {
    if (typeof obj === 'string') {
        return sanitizeString(obj);
    }
    
    if (Array.isArray(obj)) {
        return obj.map(item => sanitizeObject(item));
    }
    
    if (obj && typeof obj === 'object') {
        const sanitized: any = {};
        for (const [key, value] of Object.entries(obj)) {
            // Skip potentially dangerous keys
            if (key.startsWith('__') || key.startsWith('$')) {
                continue;
            }
            sanitized[key] = sanitizeObject(value);
        }
        return sanitized;
    }
    
    return obj;
}

// Main validation function
export function validateRequest<T>(schema: z.ZodSchema<T>) {
    return (data: any, context?: functions.https.CallableContext): T => {
        try {
            // First sanitize the input
            const sanitized = sanitizeObject(data);
            
            // Log the request for auditing (without sensitive data)
            console.log('Validating request from user:', context?.auth?.uid || 'anonymous');
            
            // Validate against schema
            const validated = schema.parse(sanitized);
            
            return validated;
            
        } catch (error) {
            if (error instanceof z.ZodError) {
                const details = error.errors.map(e => ({
                    path: e.path.join('.'),
                    message: e.message,
                    code: e.code
                }));
                
                console.error('Validation failed:', {
                    userId: context?.auth?.uid,
                    errors: details
                });
                
                throw new functions.https.HttpsError(
                    'invalid-argument',
                    `Validation failed: ${details[0].message}`,
                    { 
                        errors: details,
                        timestamp: new Date().toISOString()
                    }
                );
            }
            
            console.error('Unexpected validation error:', error);
            throw new functions.https.HttpsError(
                'internal',
                'An unexpected error occurred during validation'
            );
        }
    };
}

// Rate limiting helper
const requestCounts = new Map<string, { count: number; resetTime: number }>();

export function rateLimit(
    maxRequests: number = 10,
    windowMs: number = 60000 // 1 minute
) {
    return (context: functions.https.CallableContext) => {
        const userId = context.auth?.uid || context.rawRequest.ip || 'anonymous';
        const now = Date.now();
        
        const userLimit = requestCounts.get(userId);
        
        if (!userLimit || now > userLimit.resetTime) {
            requestCounts.set(userId, {
                count: 1,
                resetTime: now + windowMs
            });
            return;
        }
        
        if (userLimit.count >= maxRequests) {
            throw new functions.https.HttpsError(
                'resource-exhausted',
                `Rate limit exceeded. Please try again later.`,
                {
                    retryAfter: Math.ceil((userLimit.resetTime - now) / 1000)
                }
            );
        }
        
        userLimit.count++;
    };
}

// Authentication helper
export function requireAuth(context: functions.https.CallableContext) {
    if (!context.auth) {
        throw new functions.https.HttpsError(
            'unauthenticated',
            'Authentication required to access this function'
        );
    }
    
    return context.auth;
}

// Role checking helper
export async function requireRole(
    context: functions.https.CallableContext,
    requiredRoles: string[]
) {
    const auth = requireAuth(context);
    
    // Get user's custom claims
    const token = await admin.auth().getUser(auth.uid);
    const userRole = token.customClaims?.role;
    
    if (!userRole || !requiredRoles.includes(userRole)) {
        throw new functions.https.HttpsError(
            'permission-denied',
            `Insufficient permissions. Required role: ${requiredRoles.join(' or ')}`
        );
    }
    
    return userRole;
}