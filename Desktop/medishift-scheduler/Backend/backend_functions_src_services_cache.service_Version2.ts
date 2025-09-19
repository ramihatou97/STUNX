import * as admin from 'firebase-admin';

interface CacheEntry<T> {
    data: T;
    expiry: number;
    hits: number;
    created: number;
}

export class CacheService {
    private static instance: CacheService;
    private cache: Map<string, CacheEntry<any>> = new Map();
    private readonly DEFAULT_TTL = 300000; // 5 minutes
    private readonly MAX_CACHE_SIZE = 1000;
    private readonly CLEANUP_INTERVAL = 60000; // 1 minute
    
    private constructor() {
        // Start cleanup interval
        setInterval(() => this.cleanup(), this.CLEANUP_INTERVAL);
    }
    
    static getInstance(): CacheService {
        if (!CacheService.instance) {
            CacheService.instance = new CacheService();
        }
        return CacheService.instance;
    }
    
    static async get<T>(key: string): Promise<T | null> {
        return CacheService.getInstance().getItem<T>(key);
    }
    
    static async set<T>(
        key: string, 
        value: T, 
        ttl?: number
    ): Promise<void> {
        return CacheService.getInstance().setItem(key, value, ttl);
    }
    
    static async invalidate(pattern: string): Promise<void> {
        return CacheService.getInstance().invalidatePattern(pattern);
    }
    
    private async getItem<T>(key: string): Promise<T | null> {
        const cached = this.cache.get(key);
        
        if (!cached) {
            console.log(`Cache miss: ${key}`);
            return null;
        }
        
        if (cached.expiry < Date.now()) {
            console.log(`Cache expired: ${key}`);
            this.cache.delete(key);
            return null;
        }
        
        // Update hit count
        cached.hits++;
        console.log(`Cache hit: ${key} (hits: ${cached.hits})`);
        
        return cached.data as T;
    }
    
    private async setItem<T>(
        key: string, 
        value: T, 
        ttl?: number
    ): Promise<void> {
        // Check cache size limit
        if (this.cache.size >= this.MAX_CACHE_SIZE) {
            this.evictLRU();
        }
        
        this.cache.set(key, {
            data: value,
            expiry: Date.now() + (ttl || this.DEFAULT_TTL),
            hits: 0,
            created: Date.now()
        });
        
        console.log(`Cache set: ${key} (TTL: ${ttl || this.DEFAULT_TTL}ms)`);
    }
    
    private async invalidatePattern(pattern: string): Promise<void> {
        const regex = new RegExp(pattern.replace(/\*/g, '.*'));
        let invalidated = 0;
        
        for (const key of this.cache.keys()) {
            if (regex.test(key)) {
                this.cache.delete(key);
                invalidated++;
            }
        }
        
        console.log(`Cache invalidated: ${invalidated} entries matching ${pattern}`);
    }
    
    private cleanup(): void {
        const now = Date.now();
        let cleaned = 0;
        
        for (const [key, entry] of this.cache.entries()) {
            if (entry.expiry < now) {
                this.cache.delete(key);
                cleaned++;
            }
        }
        
        if (cleaned > 0) {
            console.log(`Cache cleanup: removed ${cleaned} expired entries`);
        }
    }
    
    private evictLRU(): void {
        // Find least recently used entry
        let lruKey: string | null = null;
        let minHits = Infinity;
        let oldestTime = Date.now();
        
        for (const [key, entry] of this.cache.entries()) {
            const score = entry.hits + (Date.now() - entry.created) / 10000;
            if (score < minHits) {
                minHits = score;
                lruKey = key;
            }
        }
        
        if (lruKey) {
            this.cache.delete(lruKey);
            console.log(`Cache eviction: removed ${lruKey}`);
        }
    }
    
    // Get cache statistics
    static getStats() {
        const instance = CacheService.getInstance();
        const entries = Array.from(instance.cache.entries());
        
        return {
            size: instance.cache.size,
            maxSize: instance.MAX_CACHE_SIZE,
            entries: entries.map(([key, entry]) => ({
                key,
                hits: entry.hits,
                age: Date.now() - entry.created,
                ttl: entry.expiry - Date.now()
            })),
            totalHits: entries.reduce((sum, [_, entry]) => sum + entry.hits, 0)
        };
    }
}

// Export singleton instance methods for convenience
export const cache = {
    get: CacheService.get,
    set: CacheService.set,
    invalidate: CacheService.invalidate,
    stats: CacheService.getStats
};