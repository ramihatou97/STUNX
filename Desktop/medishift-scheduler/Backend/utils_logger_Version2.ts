// Centralized Logger Module
import fs from 'fs';
import path from 'path';

type LogLevel = 'ERROR' | 'WARN' | 'INFO' | 'DEBUG';

interface LogContext {
  [key: string]: any;
}

export interface LoggerConfig {
  logLevel: string;
  outputs: {
    console: boolean;
    file: boolean;
    remoteLogging?: string;
  };
  rotation: {
    maxSize: string;
    maxFiles: number;
    compress: boolean;
  };
  redactFields: string[];
  slowQueryThreshold: number;
}

class Logger {
  private readonly logLevels: Record<LogLevel, LogLevel>;
  private logFile: string;

  constructor() {
    this.logLevels = {
      ERROR: 'ERROR',
      WARN: 'WARN',
      INFO: 'INFO',
      DEBUG: 'DEBUG'
    };
    
    // Create logs directory if it doesn't exist
    const logsDir = path.join(process.cwd(), 'logs');
    if (!fs.existsSync(logsDir)) {
      fs.mkdirSync(logsDir, { recursive: true });
    }
    
    this.logFile = path.join(logsDir, `app-${new Date().toISOString().split('T')[0]}.log`);
  }

  private formatMessage(level: LogLevel, message: string, context: LogContext = {}): string {
    return JSON.stringify({
      timestamp: new Date().toISOString(),
      level,
      message,
      context,
      pid: process.pid
    });
  }

  private writeLog(level: LogLevel, message: string, context: LogContext): void {
    const formattedMessage = this.formatMessage(level, message, context);
    
    // Console output with color coding
    const colors: Record<LogLevel, string> = {
      ERROR: '\x1b[31m', // Red
      WARN: '\x1b[33m',  // Yellow
      INFO: '\x1b[36m',  // Cyan
      DEBUG: '\x1b[37m'  // White
    };
    const reset = '\x1b[0m';
    
    console.log(`${colors[level]}[${level}]${reset} ${message}`, context);
    
    // File output
    fs.appendFileSync(this.logFile, formattedMessage + '\n');
  }

  public error(message: string, context: LogContext = {}): void {
    this.writeLog(this.logLevels.ERROR, message, context);
  }

  public warn(message: string, context: LogContext = {}): void {
    this.writeLog(this.logLevels.WARN, message, context);
  }

  public info(message: string, context: LogContext = {}): void {
    this.writeLog(this.logLevels.INFO, message, context);
  }

  public debug(message: string, context: LogContext = {}): void {
    if (process.env.NODE_ENV === 'development' || process.env.DEBUG === 'true') {
      this.writeLog(this.logLevels.DEBUG, message, context);
    }
  }

  // Log function execution with timing
  public logFunctionExecution<T extends (...args: any[]) => Promise<any>>(
    functionName: string, 
    fn: T, 
    context: LogContext = {}
  ): (...args: Parameters<T>) => Promise<ReturnType<T>> {
    return async (...args: Parameters<T>): Promise<ReturnType<T>> => {
      const startTime = Date.now();
      this.info(`Function ${functionName} started`, { ...context, args });
      
      try {
        const result = await fn(...args);
        const duration = Date.now() - startTime;
        this.info(`Function ${functionName} completed`, { 
          ...context, 
          duration: `${duration}ms`,
          success: true 
        });
        return result;
      } catch (error) {
        const duration = Date.now() - startTime;
        this.error(`Function ${functionName} failed`, { 
          ...context, 
          duration: `${duration}ms`,
          error: error instanceof Error ? error.message : String(error),
          stack: error instanceof Error ? error.stack : undefined
        });
        throw error;
      }
    };
  }
}

// Export singleton instance
export default new Logger();