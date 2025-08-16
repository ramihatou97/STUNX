import { LoggerConfig } from '../utils/logger';

const config: LoggerConfig = {
  // Log levels based on environment
  logLevel: process.env.LOG_LEVEL || (
    process.env.NODE_ENV === 'production' ? 'info' : 'debug'
  ),
  
  // Output configuration
  outputs: {
    console: process.env.LOG_TO_CONSOLE !== 'false',
    file: process.env.LOG_TO_FILE !== 'false',
    remoteLogging: process.env.REMOTE_LOG_ENDPOINT
  },
  
  // Log rotation
  rotation: {
    maxSize: '10m',  // 10 MB
    maxFiles: 30,    // Keep 30 days of logs
    compress: true   // Compress old logs
  },
  
  // Sensitive data filtering
  redactFields: ['password', 'token', 'apiKey', 'secret'],
  
  // Performance thresholds
  slowQueryThreshold: 1000, // Log queries taking > 1s as warnings
};

export default config;