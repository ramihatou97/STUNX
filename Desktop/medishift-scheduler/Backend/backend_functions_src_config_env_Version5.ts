import * as functions from 'firebase-functions';

interface EnvironmentConfig {
    firebase: {
        projectId: string;
        region: string;
    };
    services: {
        notificationServiceUrl?: string;
        mlServiceUrl?: string;
        redisUrl?: string;
    };
    features: {
        enableCache: boolean;
        enablePerformanceTracking: boolean;
        enableDetailedLogging: boolean;
    };
    limits: {
        maxResidentsPerSchedule: number;
        maxDaysPerLeaveRequest: number;
        maxConcurrentRequests: number;
    };
}

const requiredEnvVars = [
    'GCLOUD_PROJECT',
    'FUNCTION_TARGET',
    'FUNCTION_SIGNATURE_TYPE'
];

const optionalEnvVars = [
    'NOTIFICATION_SERVICE_URL',
    'ML_SERVICE_URL',
    'REDIS_URL',
    'ENABLE_CACHE',
    'ENABLE_PERFORMANCE_TRACKING',
    'ENABLE_DETAILED_LOGGING'
];

export function validateEnvironment(): void {
    const missing = requiredEnvVars.filter(
        key => !process.env[key] && !functions.config()[key.toLowerCase()]
    );
    
    if (missing.length > 0) {
        console.warn(
            `⚠️ Missing environment variables: ${missing.join(', ')}. ` +
            `Using defaults.`
        );
    }
    
    console.log('✅ Environment validated successfully');
}

export function getConfig(): EnvironmentConfig {
    const config = functions.config();
    
    return {
        firebase: {
            projectId: process.env.GCLOUD_PROJECT || config.firebase?.project_id || 'neuroman-prod',
            region: config.firebase?.region || 'us-central1'
        },
        services: {
            notificationServiceUrl: process.env.NOTIFICATION_SERVICE_URL || config.services?.notification_url,
            mlServiceUrl: process.env.ML_SERVICE_URL || config.services?.ml_url,
            redisUrl: process.env.REDIS_URL || config.services?.redis_url
        },
        features: {
            enableCache: process.env.ENABLE_CACHE === 'true' || config.features?.enable_cache === true,
            enablePerformanceTracking: process.env.ENABLE_PERFORMANCE_TRACKING !== 'false',
            enableDetailedLogging: process.env.ENABLE_DETAILED_LOGGING === 'true'
        },
        limits: {
            maxResidentsPerSchedule: parseInt(process.env.MAX_RESIDENTS || '100'),
            maxDaysPerLeaveRequest: parseInt(process.env.MAX_LEAVE_DAYS || '30'),
            maxConcurrentRequests: parseInt(process.env.MAX_CONCURRENT || '10')
        }
    };
}

export const appConfig = getConfig();