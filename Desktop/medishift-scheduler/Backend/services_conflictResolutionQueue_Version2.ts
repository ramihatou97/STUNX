import { EventEmitter } from 'events';
import Bull from 'bull';
import { Db } from 'mongodb';
import logger from '../utils/logger';
import { ScheduleVersioningService } from './scheduleVersioning';

interface ScheduleConflict {
  id: string;
  scheduleId: string;
  type: 'version_conflict' | 'time_overlap' | 'resource_conflict' | 'dependency_conflict';
  severity: 'low' | 'medium' | 'high' | 'critical';
  conflictingChanges: ConflictingChange[];
  detectedAt: Date;
  status: 'pending' | 'processing' | 'resolved' | 'failed';
  resolution?: ConflictResolution;
  metadata: any;
}

interface ConflictingChange {
  userId: string;
  change: any;
  timestamp: Date;
  version?: number;
}

interface ConflictResolution {
  strategy: 'auto_merge' | 'manual_merge' | 'priority_based' | 'timestamp_based' | 'custom';
  resolvedBy?: string;
  resolvedAt?: Date;
  result: any;
  notes?: string;
}

export class ConflictResolutionQueue extends EventEmitter {
  private queue: Bull.Queue;
  private db: Db;
  private versioningService: ScheduleVersioningService;
  private conflictHandlers: Map<string, Function>;

  constructor(db: Db, redisUrl: string) {
    super();
    this.db = db;
    this.queue = new Bull('conflict-resolution', redisUrl);
    this.versioningService = new ScheduleVersioningService(db);
    this.conflictHandlers = new Map();
    
    this.initializeQueue();
    this.registerDefaultHandlers();
  }

  private initializeQueue() {
    // Process conflicts
    this.queue.process(async (job) => {
      const conflict = job.data as ScheduleConflict;
      logger.info('Processing conflict', { 
        conflictId: conflict.id, 
        type: conflict.type 
      });

      try {
        const resolution = await this.resolveConflict(conflict);
        
        // Update conflict status
        await this.updateConflictStatus(conflict.id, 'resolved', resolution);
        
        // Emit resolution event
        this.emit('conflictResolved', { conflict, resolution });
        
        return resolution;
      } catch (error) {
        logger.error('Failed to resolve conflict', {
          conflictId: conflict.id,
          error: error.message
        });
        
        await this.updateConflictStatus(conflict.id, 'failed');
        throw error;
      }
    });

    // Handle queue events
    this.queue.on('completed', (job, result) => {
      logger.info('Conflict resolution completed', {
        jobId: job.id,
        conflictId: job.data.id
      });
    });

    this.queue.on('failed', (job, error) => {
      logger.error('Conflict resolution failed', {
        jobId: job.id,
        conflictId: job.data.id,
        error: error.message
      });
      
      // Retry logic
      if (job.attemptsMade < 3) {
        job.retry();
      }
    });
  }

  private registerDefaultHandlers() {
    // Version conflict handler
    this.registerHandler('version_conflict', async (conflict: ScheduleConflict) => {
      return await this.handleVersionConflict(conflict);
    });

    // Time overlap handler
    this.registerHandler('time_overlap', async (conflict: ScheduleConflict) => {
      return await this.handleTimeOverlap(conflict);
    });

    // Resource conflict handler
    this.registerHandler('resource_conflict', async (conflict: ScheduleConflict) => {
      return await this.handleResourceConflict(conflict);
    });

    // Dependency conflict handler
    this.registerHandler('dependency_conflict', async (conflict: ScheduleConflict) => {
      return await this.handleDependencyConflict(conflict);
    });
  }

  // Add conflict to queue
  async addConflict(conflict: Omit<ScheduleConflict, 'id' | 'detectedAt' | 'status'>): Promise<string> {
    const conflictId = this.generateConflictId();
    
    const fullConflict: ScheduleConflict = {
      ...conflict,
      id: conflictId,
      detectedAt: new Date(),
      status: 'pending'
    };

    // Save to database
    await this.db.collection('schedule_conflicts').insertOne(fullConflict);

    // Add to processing queue with priority
    const priority = this.calculatePriority(fullConflict);
    await this.queue.add(fullConflict, {
      priority,
      attempts: 3,
      backoff: {
        type: 'exponential',
        delay: 2000
      }
    });

    logger.info('Conflict added to queue', {
      conflictId,
      type: conflict.type,
      severity: conflict.severity,
      priority
    });

    return conflictId;
  }

  // Resolve conflict based on type
  private async resolveConflict(conflict: ScheduleConflict): Promise<ConflictResolution> {
    const handler = this.conflictHandlers.get(conflict.type);
    
    if (!handler) {
      throw new Error(`No handler registered for conflict type: ${conflict.type}`);
    }

    return await handler(conflict);
  }

  // Version conflict resolution
  private async handleVersionConflict(conflict: ScheduleConflict): Promise<ConflictResolution> {
    const changes = conflict.conflictingChanges;
    
    // Try auto-merge if changes don't overlap
    const canAutoMerge = this.canAutoMerge(changes);
    
    if (canAutoMerge) {
      const mergedChanges = this.mergeChanges(changes);
      
      // Apply merged changes
      await this.versioningService.createVersion(
        conflict.scheduleId,
        mergedChanges,
        'system',
        { 
          conflictResolution: true, 
          autoMerged: true 
        }
      );

      return {
        strategy: 'auto_merge',
        resolvedAt: new Date(),
        result: mergedChanges,
        notes: 'Automatically merged non-conflicting changes'
      };
    }

    // If can't auto-merge, use priority or timestamp-based resolution
    if (conflict.severity === 'critical') {
      return await this.priorityBasedResolution(conflict);
    } else {
      return await this.timestampBasedResolution(conflict);
    }
  }

  // Time overlap resolution
  private async handleTimeOverlap(conflict: ScheduleConflict): Promise<ConflictResolution> {
    const overlappingSchedules = conflict.metadata.overlappingSchedules;
    
    // Strategy: Move or split schedules
    const resolution = await this.resolveTimeOverlap(overlappingSchedules);
    
    return {
      strategy: 'custom',
      resolvedAt: new Date(),
      result: resolution,
      notes: 'Resolved time overlap by adjusting schedules'
    };
  }

  // Resource conflict resolution
  private async handleResourceConflict(conflict: ScheduleConflict): Promise<ConflictResolution> {
    const resources = conflict.metadata.resources;
    
    // Strategy: Allocate alternative resources or queue
    const resolution = await this.allocateAlternativeResources(resources);
    
    return {
      strategy: 'custom',
      resolvedAt: new Date(),
      result: resolution,
      notes: 'Allocated alternative resources'
    };
  }

  // Dependency conflict resolution
  private async handleDependencyConflict(conflict: ScheduleConflict): Promise<ConflictResolution> {
    const dependencies = conflict.metadata.dependencies;
    
    // Strategy: Reorder or adjust dependencies
    const resolution = await this.resolveDependencies(dependencies);
    
    return {
      strategy: 'custom',
      resolvedAt: new Date(),
      result: resolution,
      notes: 'Resolved dependency conflicts'
    };
  }

  // Helper methods
  private canAutoMerge(changes: ConflictingChange[]): boolean {
    const fieldMap = new Map<string, ConflictingChange[]>();
    
    for (const change of changes) {
      const field = change.change.field;
      if (!fieldMap.has(field)) {
        fieldMap.set(field, []);
      }
      fieldMap.get(field)!.push(change);
    }

    // Check if any field has multiple changes
    for (const [field, fieldChanges] of fieldMap) {
      if (fieldChanges.length > 1) {
        // Multiple users changed the same field - can't auto-merge
        return false;
      }
    }

    return true;
  }

  private mergeChanges(changes: ConflictingChange[]): any[] {
    // Sort by timestamp and merge
    const sorted = changes.sort((a, b) => 
      a.timestamp.getTime() - b.timestamp.getTime()
    );

    return sorted.map(c => c.change);
  }

  private async priorityBasedResolution(conflict: ScheduleConflict): Promise<ConflictResolution> {
    // Resolve based on user priority or role
    const prioritizedChange = conflict.conflictingChanges.sort((a, b) => {
      // Implement priority logic
      return this.getUserPriority(b.userId) - this.getUserPriority(a.userId);
    })[0];

    await this.versioningService.createVersion(
      conflict.scheduleId,
      [prioritizedChange.change],
      prioritizedChange.userId,
      { conflictResolution: true, strategy: 'priority' }
    );

    return {
      strategy: 'priority_based',
      resolvedAt: new Date(),
      result: prioritizedChange.change,
      notes: `Resolved using priority for user ${prioritizedChange.userId}`
    };
  }

  private async timestampBasedResolution(conflict: ScheduleConflict): Promise<ConflictResolution> {
    // Last write wins
    const latestChange = conflict.conflictingChanges.sort((a, b) => 
      b.timestamp.getTime() - a.timestamp.getTime()
    )[0];

    await this.versioningService.createVersion(
      conflict.scheduleId,
      [latestChange.change],
      latestChange.userId,
      { conflictResolution: true, strategy: 'timestamp' }
    );

    return {
      strategy: 'timestamp_based',
      resolvedAt: new Date(),
      result: latestChange.change,
      notes: 'Applied last write wins strategy'
    };
  }

  private async resolveTimeOverlap(overlappingSchedules: any[]): Promise<any> {
    // Implementation for resolving time overlaps
    // This could involve moving schedules, splitting them, or notifying users
    return {
      adjusted: overlappingSchedules.map(s => ({
        ...s,
        newTime: this.findAvailableTimeSlot(s)
      }))
    };
  }

  private async allocateAlternativeResources(resources: any[]): Promise<any> {
    // Find and allocate alternative resources
    return {
      allocated: resources.map(r => ({
        original: r,
        alternative: this.findAlternativeResource(r)
      }))
    };
  }

  private async resolveDependencies(dependencies: any[]): Promise<any> {
    // Topological sort or dependency resolution
    return {
      reordered: this.topologicalSort(dependencies)
    };
  }

  private calculatePriority(conflict: ScheduleConflict): number {
    const severityPriority = {
      'critical': 4,
      'high': 3,
      'medium': 2,
      'low': 1
    };

    const typePriority = {
      'dependency_conflict': 4,
      'resource_conflict': 3,
      'time_overlap': 2,
      'version_conflict': 1
    };

    return severityPriority[conflict.severity] * 10 + typePriority[conflict.type];
  }

  private getUserPriority(userId: string): number {
    // Implement user priority logic based on role, seniority, etc.
    return 1;
  }

  private findAvailableTimeSlot(schedule: any): any {
    // Logic to find next available time slot
    return schedule.time;
  }

  private findAlternativeResource(resource: any): any {
    // Logic to find alternative resource
    return resource;
  }

  private topologicalSort(dependencies: any[]): any[] {
    // Implement topological sorting for dependencies
    return dependencies;
  }

  private generateConflictId(): string {
    return `conflict_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private async updateConflictStatus(
    conflictId: string, 
    status: string, 
    resolution?: ConflictResolution
  ): Promise<void> {
    await this.db.collection('schedule_conflicts').updateOne(
      { id: conflictId },
      { 
        $set: { 
          status, 
          resolution,
          updatedAt: new Date()
        } 
      }
    );
  }

  // Register custom conflict handler
  registerHandler(type: string, handler: Function): void {
    this.conflictHandlers.set(type, handler);
    logger.info(`Registered conflict handler for type: ${type}`);
  }

  // Get conflict status
  async getConflictStatus(conflictId: string): Promise<ScheduleConflict | null> {
    return await this.db.collection<ScheduleConflict>('schedule_conflicts')
      .findOne({ id: conflictId });
  }

  // Get pending conflicts
  async getPendingConflicts(limit: number = 10): Promise<ScheduleConflict[]> {
    return await this.db.collection<ScheduleConflict>('schedule_conflicts')
      .find({ status: 'pending' })
      .limit(limit)
      .toArray();
  }
}