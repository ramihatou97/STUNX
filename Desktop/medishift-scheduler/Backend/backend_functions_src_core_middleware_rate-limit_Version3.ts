import * as functions from 'firebase-functions';
import * as admin from 'firebase-admin';

interface RateLimitConfig {
    maxRequests: number;
    windowMs: number;
}

const rateLimitStore = new Map<string, { count: number; resetTime: number }>();

export function rateLimit(config: RateLimitConfig = { maxRequests: 100, windowMs: 60000 }) {
    return (data: any, context: functions.https.CallableContext) => {
        if (!context.auth) {
            throw new functions.https.HttpsError('unauthenticated', 'Authentication required');
        }

        const userId = context.auth.uid;
        const now = Date.now();
        const userLimit = rateLimitStore.get(userId);

        if (!userLimit || userLimit.resetTime < now) {
            rateLimitStore.set(userId, {
                count: 1,
                resetTime: now + config.windowMs
            });
            return;
        }

        if (userLimit.count >= config.maxRequests) {
            throw new functions.https.HttpsError(
                'resource-exhausted',
                `Rate limit exceeded. Try again in ${Math.ceil((userLimit.resetTime - now) / 1000)} seconds`
            );
        }

        userLimit.count++;
    };
}