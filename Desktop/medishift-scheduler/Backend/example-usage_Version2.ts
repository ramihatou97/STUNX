import logger from './utils/logger';

// Example: Logging in generateMonthlySchedule function
async function generateMonthlySchedule(month: number, year: number): Promise<any[]> {
  // Log function entry
  logger.info('Generating monthly schedule', { month, year });
  
  try {
    // Your existing logic here
    const schedule = await createSchedule(month, year);
    
    // Log successful operations
    logger.debug('Schedule created', { 
      itemCount: schedule.length,
      month, 
      year 
    });
    
    return schedule;
  } catch (error) {
    // Log errors with context
    logger.error('Failed to generate monthly schedule', {
      month,
      year,
      error: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined
    });
    throw error;
  }
}

// Mock function for example
async function createSchedule(month: number, year: number): Promise<any[]> {
  return [];
}

// Wrap function with performance monitoring
const monitoredGenerateSchedule = logger.logFunctionExecution(
  'generateMonthlySchedule',
  generateMonthlySchedule,
  { module: 'scheduling' }
);

export { monitoredGenerateSchedule };