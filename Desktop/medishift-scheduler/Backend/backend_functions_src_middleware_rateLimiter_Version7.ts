import * as functions from 'firebase-functions';

interface RateLimitEntry {
    count: number;
    resetTime: number;
    violations: number;
}

const rateLimitMap = new Map<string, RateLimitEntry>();

// Clean up old entries periodically
setInterval(() => {
    const now = Date.now();
    for (const [userId, entry] of rateLimitMap.entries()) {
        if (entry.resetTime < now) {
            rateLimitMap.delete(userId);
        }
    }
}, 60000); // Clean every minute

export function rateLimit(
    maxRequests: number = 100,
    windowMs: number = 60000
) {
    return (context: functions.https.CallableContext) => {
        if (!context.auth) {
            throw new functions.https.HttpsError(
                'unauthenticated',
                'Authentication required'
            );
        }
        
        const userId = context.auth.uid;
        const now = Date.now();
        const userLimit = rateLimitMap.get(userId);
        
        if (!userLimit || userLimit.resetTime < now) {
            rateLimitMap.set(userId, {
                count: 1,
                resetTime: now + windowMs,
                violations: 0
            });
            return;
        }
        
        if (userLimit.count >= maxRequests) {
            userLimit.violations++;
            
            // Ban user temporarily if too many violations
            if (userLimit.violations > 5) {
                const banTime = windowMs * 10; // 10x the window
                userLimit.resetTime = now + banTime;
                
                throw new functions.https.HttpsError(
                    'resource-exhausted',
                    `Too many violations. Banned for ${Math.ceil(banTime / 60000)} minutes`
                );
            }
            
            throw new functions.https.HttpsError(
                'resource-exhausted',
                `Rate limit exceeded. Try again in ${
                    Math.ceil((userLimit.resetTime - now) / 1000)
                } seconds. Violations: ${userLimit.violations}/5`
            );
        }
        
        userLimit.count++;
    };
}

// Advanced rate limiting with different tiers
export function tieredRateLimit(context: functions.https.CallableContext) {
    const isAdmin = context.auth?.token?.admin === true;
    const isPremium = context.auth?.token?.premium === true;
    
    if (isAdmin) {
        return rateLimit(1000, 60000)(context); // 1000 requests per minute
    } else if (isPremium) {
        return rateLimit(200, 60000)(context); // 200 requests per minute
    } else {
        return rateLimit(50, 60000)(context); // 50 requests per minute
    }
}