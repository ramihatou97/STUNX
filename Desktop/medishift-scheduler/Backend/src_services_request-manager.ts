/**
 * Request Manager Service
 * Prevents duplicate requests and manages request lifecycle
 */

class RequestManager {
  private pendingRequests: Map<string, Promise<any>> = new Map();
  private requestTimestamps: Map<string, number> = new Map();
  private readonly MIN_REQUEST_INTERVAL = 2000; // 2 seconds minimum between identical requests
  
  /**
   * Generate a unique key for a request
   */
  private generateRequestKey(
    endpoint: string, 
    method: string, 
    params?: any
  ): string {
    const paramString = params ? JSON.stringify(params) : '';
    return `${method}:${endpoint}:${paramString}`;
  }
  
  /**
   * Check if a request is already pending
   */
  isPending(endpoint: string, method: string, params?: any): boolean {
    const key = this.generateRequestKey(endpoint, method, params);
    return this.pendingRequests.has(key);
  }
  
  /**
   * Check if a request was made too recently
   */
  isTooSoon(endpoint: string, method: string, params?: any): boolean {
    const key = this.generateRequestKey(endpoint, method, params);
    const lastTimestamp = this.requestTimestamps.get(key);
    
    if (!lastTimestamp) return false;
    
    return Date.now() - lastTimestamp < this.MIN_REQUEST_INTERVAL;
  }
  
  /**
   * Execute a request with duplicate prevention
   */
  async executeRequest<T>(
    endpoint: string,
    method: string,
    requestFn: () => Promise<T>,
    params?: any,
    options?: {
      allowDuplicate?: boolean;
      customKey?: string;
      minInterval?: number;
    }
  ): Promise<T> {
    const key = options?.customKey || this.generateRequestKey(endpoint, method, params);
    
    // Check if duplicate requests are allowed
    if (!options?.allowDuplicate) {
      // Check if request is already pending
      const pending = this.pendingRequests.get(key);
      if (pending) {
        console.log(`[RequestManager] Duplicate request prevented: ${key}`);
        return pending as Promise<T>;
      }
      
      // Check if request was made too recently
      const lastTimestamp = this.requestTimestamps.get(key);
      const minInterval = options?.minInterval || this.MIN_REQUEST_INTERVAL;
      
      if (lastTimestamp && Date.now() - lastTimestamp < minInterval) {
        const waitTime = minInterval - (Date.now() - lastTimestamp);
        throw new Error(`Please wait ${Math.ceil(waitTime / 1000)} seconds before retrying`);
      }
    }
    
    // Execute the request
    const promise = requestFn()
      .then(result => {
        // Clean up pending request
        this.pendingRequests.delete(key);
        // Update timestamp
        this.requestTimestamps.set(key, Date.now());
        return result;
      })
      .catch(error => {
        // Clean up on error
        this.pendingRequests.delete(key);
        throw error;
      });
    
    // Store as pending
    this.pendingRequests.set(key, promise);
    
    return promise;
  }
  
  /**
   * Clear all pending requests
   */
  clearPending(): void {
    this.pendingRequests.clear();
  }
  
  /**
   * Clear request history
   */
  clearHistory(): void {
    this.requestTimestamps.clear();
  }
  
  /**
   * Get pending request count
   */
  getPendingCount(): number {
    return this.pendingRequests.size;
  }
}

export const requestManager = new RequestManager();