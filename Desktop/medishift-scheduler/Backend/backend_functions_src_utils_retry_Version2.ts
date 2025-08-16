export interface RetryOptions {
    maxRetries?: number;
    baseDelay?: number;
    maxDelay?: number;
    backoffMultiplier?: number;
    shouldRetry?: (error: Error) => boolean;
}

export async function retryWithBackoff<T>(
    fn: () => Promise<T>,
    options: RetryOptions = {}
): Promise<T> {
    const {
        maxRetries = 3,
        baseDelay = 1000,
        maxDelay = 30000,
        backoffMultiplier = 2,
        shouldRetry = () => true
    } = options;
    
    let lastError: Error;
    
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error as Error;
            
            // Check if we should retry this error
            if (!shouldRetry(lastError)) {
                throw lastError;
            }
            
            if (attempt < maxRetries - 1) {
                const delay = Math.min(
                    baseDelay * Math.pow(backoffMultiplier, attempt),
                    maxDelay
                );
                
                // Add jitter to prevent thundering herd
                const jitter = Math.random() * delay * 0.1;
                const finalDelay = delay + jitter;
                
                console.log(
                    `Retry ${attempt + 1}/${maxRetries} after ${Math.round(finalDelay)}ms. ` +
                    `Error: ${lastError.message}`
                );
                
                await new Promise(resolve => setTimeout(resolve, finalDelay));
            }
        }
    }
    
    console.error(`All ${maxRetries} retries failed:`, lastError!);
    throw lastError!;
}

// Specialized retry for network errors
export async function retryNetworkRequest<T>(
    fn: () => Promise<T>
): Promise<T> {
    return retryWithBackoff(fn, {
        maxRetries: 5,
        baseDelay: 500,
        shouldRetry: (error) => {
            const message = error.message.toLowerCase();
            return message.includes('network') ||
                   message.includes('timeout') ||
                   message.includes('econnrefused') ||
                   message.includes('unavailable');
        }
    });
}

// Specialized retry for Firestore operations
export async function retryFirestoreOperation<T>(
    fn: () => Promise<T>
): Promise<T> {
    return retryWithBackoff(fn, {
        maxRetries: 3,
        baseDelay: 200,
        shouldRetry: (error: any) => {
            // Retry on specific Firestore errors
            const code = error.code;
            return code === 'unavailable' ||
                   code === 'deadline-exceeded' ||
                   code === 'resource-exhausted' ||
                   code === 'internal';
        }
    });
}