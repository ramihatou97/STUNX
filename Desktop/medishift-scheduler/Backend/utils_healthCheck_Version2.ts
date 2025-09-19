import fs from 'fs/promises';
import path from 'path';
import os from 'os';
import logger from './logger';

// Simple configurable circuit breaker to protect flaky or slow health checks
class CircuitBreaker {
  private failures = 0;
  private lastFailTime = 0;
  private state: 'CLOSED' | 'OPEN' | 'HALF_OPEN' = 'CLOSED';
  private readonly threshold: number;
  private readonly timeout: number; // milliseconds

  constructor(threshold = 5, timeout = 60000) {
    this.threshold = threshold;
    this.timeout = timeout;
  }

  public getState(): 'CLOSED' | 'OPEN' | 'HALF_OPEN' {
    // If OPEN and timeout expired, report HALF_OPEN to caller
    if (this.state === 'OPEN' && Date.now() - this.lastFailTime > this.timeout) {
      return 'HALF_OPEN';
    }
    return this.state;
  }

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    // Evaluate current state before executing
    if (this.state === 'OPEN') {
      if (Date.now() - this.lastFailTime > this.timeout) {
        this.state = 'HALF_OPEN';
        logger.info('Circuit breaker moving to HALF_OPEN state');
      } else {
        throw new Error('Circuit breaker is OPEN');
      }
    }

    try {
      const result = await fn();
      // Success while HALF_OPEN closes the circuit
      if (this.state === 'HALF_OPEN') {
        this.state = 'CLOSED';
        this.failures = 0;
        logger.info('Circuit breaker CLOSED after successful trial');
      }
      return result;
    } catch (err) {
      this.failures++;
      this.lastFailTime = Date.now();
      logger.warn('Circuit breaker observed a failure', { failures: this.failures });
      if (this.failures >= this.threshold) {
        this.state = 'OPEN';
        logger.error('Circuit breaker OPENED due to repeated failures', { threshold: this.threshold });
      }
      throw err;
    }
  }
}

export interface HealthCheckResult {
  status: 'healthy' | 'degraded' | 'unhealthy';
  details?: Record<string, any>;
  error?: string;
  duration?: string;
  timestamp?: string;
  consecutiveFailures?: number;
}

export interface HealthCheckOptions {
  critical?: boolean;
  timeout?: number;
  interval?: number;
  maxFailures?: number;
  // Circuit breaker configuration
  breakerThreshold?: number;
  breakerTimeout?: number;
}

export interface HealthCheck {
  name: string;
  checkFunction: () => Promise<HealthCheckResult>;
  critical: boolean;
  timeout: number;
  interval: number;
  lastCheck: string | null;
  lastResult: HealthCheckResult | null;
  consecutiveFailures: number;
  maxFailures: number;
  // Optional circuit breaker protecting this check
  circuitBreaker?: { execute: <T>(fn: () => Promise<T>) => Promise<T> };
}

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  healthScore: string;
  checks: Record<string, HealthCheckResult>;
  summary: {
    total: number;
    healthy: number;
    unhealthy: number;
  };
}

export interface SystemMetrics {
  uptime: number;
  timestamp: string;
  process: {
    pid: number;
    version: string;
    platform: string;
    arch: string;
  };
  memory: NodeJS.MemoryUsage;
  system: {
    hostname: string;
    type: string;
    release: string;
    totalMemory: number;
    freeMemory: number;
    cpus: number;
    loadAverage: number[];
  };
  checks: Record<string, HealthCheckResult | null>;
}

class HealthMonitor {
  public checks: Map<string, HealthCheck>;
  private healthHistory: SystemHealth[];
  private maxHistorySize: number;
  private startTime: number;

  constructor() {
    this.checks = new Map<string, HealthCheck>();
    this.healthHistory = [];
    this.maxHistorySize = 100;
    this.startTime = Date.now();
    
    // Register default system checks
    this.registerSystemChecks();
  }

  // Register a custom health check
  public registerCheck(
    name: string, 
    checkFunction: () => Promise<HealthCheckResult>, 
    options: HealthCheckOptions = {}
  ): void {
    const maxFailures = options.maxFailures || 3;
    const breakerThreshold = options.breakerThreshold || maxFailures || 5;
    const breakerTimeout = options.breakerTimeout || 60000;

    this.checks.set(name, {
      name,
      checkFunction,
      critical: options.critical || false,
      timeout: options.timeout || 5000,
      interval: options.interval || 60000, // Default: check every minute
      lastCheck: null,
      lastResult: null,
      consecutiveFailures: 0,
      maxFailures,
      circuitBreaker: new CircuitBreaker(breakerThreshold, breakerTimeout)
    });
    
    logger.info('Health check registered', { name, options });
  }


  // Register default system checks
  private registerSystemChecks(): void {
    // Memory usage check
    this.registerCheck('memory', async () => {
      const used = process.memoryUsage();
      const totalMem = os.totalmem();
      const freeMem = os.freemem();
      const usagePercent = ((totalMem - freeMem) / totalMem) * 100;
      
      return {
        status: usagePercent < 90 ? 'healthy' : 'degraded',
        details: {
          heapUsed: `${Math.round(used.heapUsed / 1024 / 1024)} MB`,
          heapTotal: `${Math.round(used.heapTotal / 1024 / 1024)} MB`,
          rss: `${Math.round(used.rss / 1024 / 1024)} MB`,
          external: `${Math.round(used.external / 1024 / 1024)} MB`,
          systemMemoryUsage: `${usagePercent.toFixed(2)}%`,
          freeMemory: `${Math.round(freeMem / 1024 / 1024)} MB`
        }
      };
    }, { critical: true });

    // CPU usage check
    this.registerCheck('cpu', async () => {
      const cpus = os.cpus();
      const loadAvg = os.loadavg();
      
      return {
        status: loadAvg[0] < cpus.length * 0.8 ? 'healthy' : 'degraded',
        details: {
          cores: cpus.length,
          loadAverage: {
            '1min': loadAvg[0].toFixed(2),
            '5min': loadAvg[1].toFixed(2),
            '15min': loadAvg[2].toFixed(2)
          },
          model: cpus[0].model
        }
      };
    });

    // Disk space check
    this.registerCheck('diskSpace', async () => {
      try {
        const stats = await fs.stat(process.cwd());
        // This is a simplified check - in production, use a proper disk space library
        return {
          status: 'healthy',
          details: {
            workingDirectory: process.cwd(),
            accessible: true
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
    });

    // Process uptime check
    this.registerCheck('uptime', async () => {
      const uptime = process.uptime();
      const systemUptime = os.uptime();
      
      return {
        status: 'healthy',
        details: {
          processUptime: this.formatUptime(uptime),
          systemUptime: this.formatUptime(systemUptime),
          startTime: new Date(this.startTime).toISOString()
        }
      };
    });
  }

  // Format uptime to human readable
  private formatUptime(seconds: number): string {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    return `${days}d ${hours}h ${minutes}m ${secs}s`;
  }

  // Execute a single health check with timeout
  public async executeCheck(check: HealthCheck): Promise<HealthCheckResult> {
    const startTime = Date.now();
    
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error('Health check timeout')), check.timeout);
    });

    const runner = async () => {
      return await Promise.race([
        check.checkFunction(),
        timeoutPromise
      ]);
    };

    try {
      let resultRaw: HealthCheckResult;

      // If a circuit breaker is attached, use it to execute; it may throw if OPEN
      if (check.circuitBreaker) {
        try {
          resultRaw = await check.circuitBreaker.execute<HealthCheckResult>(runner);
        } catch (cbErr) {
          // Circuit breaker prevented execution (OPEN) or runner failed inside breaker
          if (cbErr instanceof Error && cbErr.message === 'Circuit breaker is OPEN') {
            const duration = Date.now() - startTime;
            check.consecutiveFailures++;

            const openResult: HealthCheckResult = {
              status: 'degraded',
              error: 'circuit breaker OPEN — check suspended',
              duration: `${duration}ms`,
              timestamp: new Date().toISOString(),
              consecutiveFailures: check.consecutiveFailures
            };

            check.lastCheck = openResult.timestamp ?? new Date().toISOString();
            check.lastResult = openResult;

            logger.warn(`Health check skipped by circuit breaker: ${check.name}`, {
              consecutiveFailures: check.consecutiveFailures
            });

            return openResult;
          }

          // rethrow other errors to be handled below
          throw cbErr;
        }
      } else {
        resultRaw = await runner();
      }

      const duration = Date.now() - startTime;

      check.lastCheck = new Date().toISOString();
      check.lastResult = {
        ...resultRaw,
        duration: `${duration}ms`,
        timestamp: check.lastCheck
      };
      check.consecutiveFailures = 0;

      return check.lastResult;
    } catch (error) {
      const duration = Date.now() - startTime;
      check.consecutiveFailures++;

      const result: HealthCheckResult = {
        status: 'unhealthy',
        error: error instanceof Error ? error.message : String(error),
        duration: `${duration}ms`,
        timestamp: new Date().toISOString(),
        consecutiveFailures: check.consecutiveFailures
      };

  check.lastCheck = result.timestamp ?? new Date().toISOString();
  check.lastResult = result;

      logger.error(`Health check failed: ${check.name}`, {
        error: error instanceof Error ? error.message : String(error),
        consecutiveFailures: check.consecutiveFailures
      });

      return result;
    }
  }

  // Run all health checks
  public async runAllChecks(): Promise<SystemHealth> {
    const results: Record<string, HealthCheckResult> = {};
    const promises: Promise<void>[] = [];
    
    for (const [name, check] of this.checks) {
      promises.push(
        this.executeCheck(check).then(result => {
          results[name] = result;
        })
      );
    }
    
    await Promise.allSettled(promises);
    
    // Calculate overall health
    const overallHealth = this.calculateOverallHealth(results);
    
    // Store in history
    this.addToHistory(overallHealth);
    
    return overallHealth;
  }

  // Calculate overall system health
  private calculateOverallHealth(checkResults: Record<string, HealthCheckResult>): SystemHealth {
    let overallStatus: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
    let healthyChecks = 0;
    let totalChecks = 0;
    let criticalFailure = false;
    
    for (const [name, result] of Object.entries(checkResults)) {
      totalChecks++;
      
      if (result.status === 'healthy') {
        healthyChecks++;
      } else if (result.status === 'unhealthy') {
        const check = this.checks.get(name);
        if (check && check.critical) {
          criticalFailure = true;
        }
      }
    }
    
    if (criticalFailure) {
      overallStatus = 'unhealthy';
    } else if (healthyChecks < totalChecks) {
      overallStatus = 'degraded';
    }
    
    const healthScore = totalChecks > 0 ? (healthyChecks / totalChecks) * 100 : 0;
    
    return {
      status: overallStatus,
      timestamp: new Date().toISOString(),
      healthScore: `${healthScore.toFixed(1)}%`,
      checks: checkResults,
      summary: {
        total: totalChecks,
        healthy: healthyChecks,
        unhealthy: totalChecks - healthyChecks
      }
    };
  }

  // Add result to history
  private addToHistory(result: SystemHealth): void {
    this.healthHistory.push(result);
    
    // Keep history size limited
    if (this.healthHistory.length > this.maxHistorySize) {
      this.healthHistory.shift();
    }
  }

  // Get health history
  public getHistory(limit: number = 10): SystemHealth[] {
    return this.healthHistory.slice(-limit);
  }

  // Start automated health monitoring
  public startMonitoring(): void {
    logger.info('Starting health monitoring');
    
    // Run initial check
    this.runAllChecks();
    
    // Schedule periodic checks
    for (const [name, check] of this.checks) {
      setInterval(async () => {
        await this.executeCheck(check);
        
        // Alert on critical failures
        if (check.critical && check.consecutiveFailures >= check.maxFailures) {
          this.handleCriticalFailure(name, check);
        }
      }, check.interval);
    }
    
    // Overall health check every minute
    setInterval(() => {
      this.runAllChecks();
    }, 60000);
  }

  // Handle critical failures
  private handleCriticalFailure(name: string, check: HealthCheck): void {
    logger.error(`CRITICAL: Health check ${name} has failed ${check.consecutiveFailures} times`, {
      checkName: name,
      lastResult: check.lastResult,
      maxFailures: check.maxFailures
    });
    
    // Here you could trigger alerts, send emails, etc.
    // Example: this.sendAlert(name, check);
  }

  // Get current health status
  public async getStatus(): Promise<SystemHealth> {
    return await this.runAllChecks();
  }

  // Get detailed metrics
  public async getMetrics(): Promise<SystemMetrics> {
    const metrics: SystemMetrics = {
      uptime: process.uptime(),
      timestamp: new Date().toISOString(),
      process: {
        pid: process.pid,
        version: process.version,
        platform: process.platform,
        arch: process.arch
      },
      memory: process.memoryUsage(),
      system: {
        hostname: os.hostname(),
        type: os.type(),
        release: os.release(),
        totalMemory: os.totalmem(),
        freeMemory: os.freemem(),
        cpus: os.cpus().length,
        loadAverage: os.loadavg()
      },
      checks: {}
    };
    
    // Add all check results
    for (const [name, check] of this.checks) {
      metrics.checks[name] = check.lastResult;
    }
    
    return metrics;
  }
}

export default new HealthMonitor();