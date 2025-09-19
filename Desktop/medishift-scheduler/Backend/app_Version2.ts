import express, { Application } from 'express';
import healthMonitor from './utils/healthCheck';
import healthRoutes from './routes/health';
import logger from './utils/logger';
import {
  registerDatabaseCheck,
  registerAPICheck,
  registerFileSystemCheck
} from './healthChecks/appHealthChecks';

const app: Application = express();

// Initialize health monitoring
async function initializeHealthMonitoring(): Promise<void> {
  logger.info('Initializing health monitoring');
  
  // Register your app-specific checks
  // Example: Register database check
  // registerDatabaseCheck(mongoClient);
  
  // Example: Register external API checks
  // registerAPICheck('payment_gateway', 'https://api.payment.com/health');
  
  // Register file system checks
  registerFileSystemCheck([
    './logs',
    './uploads',
    './temp'
  ]);
  
  // Custom business logic check
  healthMonitor.registerCheck('business_logic', async () => {
    try {
      // Check if your critical business functions are working
      // For example, check if schedule generation is working
      const testDate = new Date();
      // const canGenerateSchedule = await testScheduleGeneration(testDate);
      
      return {
        status: 'healthy',
        details: {
          lastCheck: new Date().toISOString(),
          // Add your business-specific metrics
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
  }, { critical: true });
  
  // Start automated monitoring
  healthMonitor.startMonitoring();
  
  logger.info('Health monitoring initialized successfully');
}

// Mount health routes
app.use('/api', healthRoutes);

// Initialize monitoring on app start
initializeHealthMonitoring().catch(error => {
  logger.error('Failed to initialize health monitoring', { 
    error: error instanceof Error ? error.message : String(error)
  });
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, performing graceful shutdown');
  // Add cleanup logic here
  process.exit(0);
});

export default app;