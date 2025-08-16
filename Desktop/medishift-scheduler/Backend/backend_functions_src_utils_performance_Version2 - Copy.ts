import * as admin from 'firebase-admin';

interface PerformanceMetric {
    operation: string;
    duration: number;
    timestamp: number;
    success: boolean;
    error?: string;
}

export class PerformanceTracker {
    private static metrics = new Map<string, number[]>();
    private static errors = new Map<string, number>();
    
    static async track<T>(
        operation: string,
        fn: () => Promise<T>
    ): Promise<T> {
        const start = Date.now();
        const metric: PerformanceMetric = {
            operation,
            duration: 0,
            timestamp: start,
            success: false
        };
        
        try {
            const result = await fn();
            const duration = Date.now() - start;
            
            metric.duration = duration;
            metric.success = true;
            
            // Store metric
            if (!this.metrics.has(operation)) {
                this.metrics.set(operation, []);
            }
            this.metrics.get(operation)!.push(duration);
            
            // Keep only last 1000 metrics per operation
            const metrics = this.metrics.get(operation)!;
            if (metrics.length > 1000) {
                metrics.shift();
            }
            
            // Log if slow
            if (duration > 3000) {
                console.warn(`⚠️ Slow operation: ${operation} took ${duration}ms`);
                await this.logSlowOperation(metric);
            }
            
            // Log to Firestore periodically (every 100th call)
            if (metrics.length % 100 === 0) {
                await this.saveMetricsToFirestore(operation);
            }
            
            return result;
        } catch (error: any) {
            const duration = Date.now() - start;
            
            metric.duration = duration;
            metric.success = false;
            metric.error = error.message;
            
            // Track errors
            const errorCount = (this.errors.get(operation) || 0) + 1;
            this.errors.set(operation, errorCount);
            
            console.error(`❌ ${operation} failed after ${duration}ms:`, error);
            await this.logError(metric);
            
            throw error;
        }
    }
    
    static getStats(operation: string) {
        const times = this.metrics.get(operation) || [];
        const errorCount = this.errors.get(operation) || 0;
        
        if (times.length === 0) return null;
        
        const sorted = [...times].sort((a, b) => a - b);
        
        return {
            count: times.length,
            errors: errorCount,
            errorRate: errorCount / (times.length + errorCount),
            avg: times.reduce((a, b) => a + b, 0) / times.length,
            min: Math.min(...times),
            max: Math.max(...times),
            median: sorted[Math.floor(sorted.length / 2)],
            p95: this.percentile(sorted, 0.95),
            p99: this.percentile(sorted, 0.99),
            stdDev: this.standardDeviation(times)
        };
    }
    
    private static percentile(sortedArr: number[], p: number): number {
        const index = Math.ceil(sortedArr.length * p) - 1;
        return sortedArr[Math.max(0, index)];
    }
    
    private static standardDeviation(arr: number[]): number {
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const squaredDiffs = arr.map(x => Math.pow(x - mean, 2));
        const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / arr.length;
        return Math.sqrt(avgSquaredDiff);
    }
    
    private static async saveMetricsToFirestore(operation: string) {
        try {
            const stats = this.getStats(operation);
            if (!stats) return;
            
            await admin.firestore()
                .collection('performanceMetrics')
                .add({
                    operation,
                    stats,
                    timestamp: admin.firestore.FieldValue.serverTimestamp()
                });
        } catch (error) {
            console.error('Failed to save metrics:', error);
        }
    }
    
    private static async logSlowOperation(metric: PerformanceMetric) {
        try {
            await admin.firestore()
                .collection('slowOperations')
                .add({
                    ...metric,
                    timestamp: admin.firestore.Timestamp.fromMillis(metric.timestamp)
                });
        } catch (error) {
            console.error('Failed to log slow operation:', error);
        }
    }
    
    private static async logError(metric: PerformanceMetric) {
        try {
            await admin.firestore()
                .collection('operationErrors')
                .add({
                    ...metric,
                    timestamp: admin.firestore.Timestamp.fromMillis(metric.timestamp)
                });
        } catch (error) {
            console.error('Failed to log error:', error);
        }
    }
    
    static clearMetrics() {
        this.metrics.clear();
        this.errors.clear();
    }
}