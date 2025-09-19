import express, { Request, Response } from 'express';
import healthMonitor from '../utils/healthCheck';
import logger from '../utils/logger';

const router = express.Router();

// Basic health check - for load balancers
router.get('/health', async (req: Request, res: Response) => {
  try {
    const health = await healthMonitor.getStatus();
    const statusCode = health.status === 'healthy' ? 200 : 
                       health.status === 'degraded' ? 503 : 500;
    
    res.status(statusCode).json({
      status: health.status,
      timestamp: health.timestamp,
      healthScore: health.healthScore
    });
  } catch (error) {
    logger.error('Health check failed', { 
      error: error instanceof Error ? error.message : String(error) 
    });
    res.status(500).json({
      status: 'error',
      message: 'Health check failed'
    });
  }
});

// Detailed health check - for monitoring systems
router.get('/health/detailed', async (req: Request, res: Response) => {
  try {
    const health = await healthMonitor.getStatus();
    res.json(health);
  } catch (error) {
    logger.error('Detailed health check failed', { 
      error: error instanceof Error ? error.message : String(error)
    });
    res.status(500).json({
      status: 'error',
      message: 'Health check failed',
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

// Live check - immediate response
router.get('/health/live', (req: Request, res: Response) => {
  res.json({
    status: 'alive',
    timestamp: new Date().toISOString()
  });
});

// Ready check - checks if app is ready to serve traffic
router.get('/health/ready', async (req: Request, res: Response) => {
  try {
    const health = await healthMonitor.getStatus();
    
    // Check if critical services are healthy
    const criticalHealthy = Object.entries(health.checks)
      .filter(([name, result]) => {
        const check = healthMonitor.checks.get(name);
        return check && check.critical;
      })
      .every(([_, result]) => result.status === 'healthy');
    
    if (criticalHealthy) {
      res.json({
        status: 'ready',
        timestamp: new Date().toISOString()
      });
    } else {
      res.status(503).json({
        status: 'not_ready',
        timestamp: new Date().toISOString(),
        reason: 'Critical services unhealthy'
      });
    }
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error instanceof Error ? error.message : String(error)
    });
  }
});

// Metrics endpoint
router.get('/health/metrics', async (req: Request, res: Response) => {
  try {
    const metrics = await healthMonitor.getMetrics();
    res.json(metrics);
  } catch (error) {
    logger.error('Failed to get metrics', { 
      error: error instanceof Error ? error.message : String(error)
    });
    res.status(500).json({
      status: 'error',
      message: 'Failed to retrieve metrics'
    });
  }
});

// Health history
router.get('/health/history', (req: Request, res: Response) => {
  const limit = parseInt(req.query.limit as string) || 10;
  const history = healthMonitor.getHistory(limit);
  res.json({
    history,
    count: history.length
  });
});

export default router;