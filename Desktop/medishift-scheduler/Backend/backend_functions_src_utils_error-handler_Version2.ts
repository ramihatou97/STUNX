import * as functions from 'firebase-functions';

export class AppError extends Error {
    constructor(
        public code: string,
        public message: string,
        public statusCode: number = 500
    ) {
        super(message);
        this.name = 'AppError';
    }
}

export function wrapAsync(fn: Function) {
    return async (...args: any[]) => {
        try {
            return await fn(...args);
        } catch (error: any) {
            console.error(`Function error: ${error.message}`, error);
            
            if (error instanceof AppError) {
                throw new functions.https.HttpsError(
                    error.code as any,
                    error.message
                );
            }
            
            if (error.code && error.message) {
                throw new functions.https.HttpsError(
                    error.code,
                    error.message
                );
            }
            
            throw new functions.https.HttpsError(
                'internal',
                'An unexpected error occurred'
            );
        }
    };
}