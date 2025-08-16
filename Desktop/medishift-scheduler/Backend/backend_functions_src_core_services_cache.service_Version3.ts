import * as admin from 'firebase-admin';

interface CacheEntry<T> {
    data: T;
    expiry: number;
}

export class CacheService {
    private static instance: CacheService;
    private cache: Map<string, CacheEntry<any>> = new Map();
    private readonly DEFAULT_TTL = 300000; // 5 minutes

    static getInstance(): CacheService {
        if (!CacheService.instance) {
            CacheService.instance = new CacheService();
        }
        return CacheService.instance;
    }

    async get<T>(key: string, fetcher: () => Promise<T>, ttl?: number): Promise<T> {
        const cached = this.cache.get(key);
        
        if (cached && cached.expiry > Date.now()) {
            console.log(`Cache hit: ${key}`);
            return cached.data as T;
        }

        console.log(`Cache miss: ${key}`);
        const data = await fetcher();
        
        this.cache.set(key, {
            data,
            expiry: Date.now() + (ttl || this.DEFAULT_TTL)
        });

        return data;
    }

    invalidate(pattern?: string): void {
        if (!pattern) {
            this.cache.clear();
            return;
        }

        for (const key of this.cache.keys()) {
            if (key.includes(pattern)) {
                this.cache.delete(key);
            }
        }
    }
}