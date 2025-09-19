import healthMonitor from '../utils/healthCheck';
import logger from '../utils/logger';
import { Db } from 'mongodb';
import { RedisClientType } from 'redis';

// Database health check
export function registerDatabaseCheck(db: Db): void {
  healthMonitor.registerCheck('database', async () => {
    try {
      // Example for MongoDB
      const startTime = Date.now();
      await db.admin().ping();
      const responseTime = Date.now() - startTime;
      
      return {
        status: responseTime < 1000 ? 'healthy' : 'degraded',
        details: {
          responseTime: `${responseTime}ms`,
          connected: true
        }
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        details: {
          error: error instanceof Error ? error.message : String(error),
          connected: false
        }
      };
    }
  }, { critical: true, timeout: 5000 });
}

// External API health check
export function registerAPICheck(apiName: string, apiUrl: string): void {
  healthMonitor.registerCheck(`api_${apiName}`, async () => {
    try {
      const startTime = Date.now();
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);
      
      const response = await fetch(apiUrl, {
        method: 'HEAD',
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      
      const responseTime = Date.now() - startTime;
      
      return {
        status: response.ok ? 'healthy' : 'degraded',
        details: {
          statusCode: response.status,
          responseTime: `${responseTime}ms`,
          url: apiUrl
        }
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        details: {
          error: error instanceof Error ? error.message : String(error),
          url: apiUrl
        }
      };
    }
  }, { critical: false, timeout: 5000 });
}

// Cache health check (Redis example)
export function registerCacheCheck(redisClient: RedisClientType): void {
  healthMonitor.registerCheck('cache', async () => {
    try {
      const startTime = Date.now();
      await redisClient.ping();
      const responseTime = Date.now() - startTime;
      
      return {
        status: responseTime < 100 ? 'healthy' : 'degraded',
        details: {
          responseTime: `${responseTime}ms`,
          connected: redisClient.isReady
        }
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        details: {
          error: error instanceof Error ? error.message : String(error),
          connected: false
        }
      };
    }
  }, { critical: false });
}

// Queue health check
interface QueueStats {
  pending: number;
  processing: number;
  completed: number;
  failed: number;
}

export function registerQueueCheck(queueName: string, checkFunction: () => Promise<QueueStats>): void {
  healthMonitor.registerCheck(`queue_${queueName}`, async () => {
    try {
      const queueStats = await checkFunction();
      const isHealthy = queueStats.pending < 1000 && queueStats.failed < 100;
      
      return {
        status: isHealthy ? 'healthy' : 'degraded',
        details: {
          pending: queueStats.pending,
          processing: queueStats.processing,
          completed: queueStats.completed,
          failed: queueStats.failed
        }
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        details: {
          error: error instanceof Error ? error.message : String(error)
        }
      };
    }
  }, { critical: false });
}

// File system check
export function registerFileSystemCheck(directories: string[]): void {
  healthMonitor.registerCheck('filesystem', async () => {
    const fs = require('fs').promises;
    const results: Record<string, {
      accessible: boolean;
      readable?: boolean;
      writable?: boolean;
      error?: string;
    }> = {};
    let allHealthy = true;
    
    for (const dir of directories) {
      try {
        await fs.access(dir, fs.constants.R_OK | fs.constants.W_OK);
        results[dir] = { accessible: true, readable: true, writable: true };
      } catch (error) {
        results[dir] = { 
          accessible: false, 
          error: error instanceof Error ? error.message : String(error)
        };
        allHealthy = false;
      }
    }
    
    return {
      status: allHealthy ? 'healthy' : 'unhealthy',
      details: {
        directories: results
      }
    };
  }, { critical: true });
}