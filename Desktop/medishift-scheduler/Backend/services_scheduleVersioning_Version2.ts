import { v4 as uuidv4 } from 'uuid';
import { Db, Collection } from 'mongodb';
import logger from '../utils/logger';

interface ScheduleVersion {
  versionId: string;
  scheduleId: string;
  version: number;
  changes: ScheduleChange[];
  previousVersionId?: string;
  createdBy: string;
  createdAt: Date;
  isActive: boolean;
  metadata: {
    reason?: string;
    conflictResolution?: boolean;
    autoMerged?: boolean;
  };
}

interface ScheduleChange {
  field: string;
  oldValue: any;
  newValue: any;
  timestamp: Date;
  changedBy: string;
}

interface Schedule {
  id: string;
  currentVersion: number;
  versionHistory: string[];
  data: any;
  lastModified: Date;
  lastModifiedBy: string;
  locked?: boolean;
  lockedBy?: string;
  lockedAt?: Date;
}

export class ScheduleVersioningService {
  private db: Db;
  private schedulesCollection: Collection<Schedule>;
  private versionsCollection: Collection<ScheduleVersion>;

  constructor(db: Db) {
    this.db = db;
    this.schedulesCollection = db.collection<Schedule>('schedules');
    this.versionsCollection = db.collection<ScheduleVersion>('schedule_versions');
    
    // Create indexes for better performance
    this.createIndexes();
  }

  private async createIndexes() {
    await this.schedulesCollection.createIndex({ id: 1, currentVersion: 1 });
    await this.versionsCollection.createIndex({ scheduleId: 1, version: 1 });
    await this.versionsCollection.createIndex({ versionId: 1 });
    await this.versionsCollection.createIndex({ createdAt: -1 });
  }

  // Create a new version of a schedule
  async createVersion(
    scheduleId: string, 
    changes: ScheduleChange[], 
    userId: string,
    metadata: any = {}
  ): Promise<ScheduleVersion> {
    try {
      // Get current schedule
      const schedule = await this.schedulesCollection.findOne({ id: scheduleId });
      
      if (!schedule) {
        throw new Error(`Schedule ${scheduleId} not found`);
      }

      // Check if schedule is locked by another user
      if (schedule.locked && schedule.lockedBy !== userId) {
        throw new Error(`Schedule is locked by user ${schedule.lockedBy}`);
      }

      const newVersion: ScheduleVersion = {
        versionId: uuidv4(),
        scheduleId,
        version: schedule.currentVersion + 1,
        changes,
        previousVersionId: schedule.versionHistory[schedule.versionHistory.length - 1],
        createdBy: userId,
        createdAt: new Date(),
        isActive: true,
        metadata
      };

      // Save the new version
      await this.versionsCollection.insertOne(newVersion);

      // Update the schedule with new version
      await this.schedulesCollection.updateOne(
        { id: scheduleId },
        {
          $set: {
            currentVersion: newVersion.version,
            lastModified: new Date(),
            lastModifiedBy: userId
          },
          $push: { versionHistory: newVersion.versionId }
        }
      );

      logger.info('Schedule version created', {
        scheduleId,
        versionId: newVersion.versionId,
        version: newVersion.version,
        userId
      });

      return newVersion;
    } catch (error) {
      logger.error('Failed to create schedule version', {
        scheduleId,
        userId,
        error: error.message
      });
      throw error;
    }
  }

  // Get version history for a schedule
  async getVersionHistory(scheduleId: string, limit: number = 10): Promise<ScheduleVersion[]> {
    const versions = await this.versionsCollection
      .find({ scheduleId })
      .sort({ version: -1 })
      .limit(limit)
      .toArray();

    return versions;
  }

  // Rollback to a specific version
  async rollbackToVersion(
    scheduleId: string, 
    targetVersionId: string, 
    userId: string
  ): Promise<Schedule> {
    try {
      const targetVersion = await this.versionsCollection.findOne({ 
        versionId: targetVersionId 
      });

      if (!targetVersion) {
        throw new Error(`Version ${targetVersionId} not found`);
      }

      // Create a rollback version
      const rollbackChanges: ScheduleChange[] = [{
        field: 'rollback',
        oldValue: null,
        newValue: targetVersionId,
        timestamp: new Date(),
        changedBy: userId
      }];

      await this.createVersion(
        scheduleId, 
        rollbackChanges, 
        userId,
        { 
          reason: 'rollback', 
          rollbackTo: targetVersionId 
        }
      );

      // Apply the rollback
      const schedule = await this.applyVersion(scheduleId, targetVersionId);

      logger.info('Schedule rolled back', {
        scheduleId,
        targetVersionId,
        userId
      });

      return schedule;
    } catch (error) {
      logger.error('Failed to rollback schedule', {
        scheduleId,
        targetVersionId,
        error: error.message
      });
      throw error;
    }
  }

  // Apply a specific version to a schedule
  private async applyVersion(scheduleId: string, versionId: string): Promise<Schedule> {
    const version = await this.versionsCollection.findOne({ versionId });
    
    if (!version) {
      throw new Error(`Version ${versionId} not found`);
    }

    // Reconstruct schedule data from version history
    const allVersions = await this.versionsCollection
      .find({ 
        scheduleId, 
        version: { $lte: version.version } 
      })
      .sort({ version: 1 })
      .toArray();

    let scheduleData = {};
    
    // Apply all changes up to target version
    for (const v of allVersions) {
      for (const change of v.changes) {
        scheduleData[change.field] = change.newValue;
      }
    }

    const updatedSchedule = await this.schedulesCollection.findOneAndUpdate(
      { id: scheduleId },
      { 
        $set: { 
          data: scheduleData,
          currentVersion: version.version,
          lastModified: new Date()
        } 
      },
      { returnDocument: 'after' }
    );

    return updatedSchedule.value!;
  }

  // Compare two versions
  async compareVersions(versionId1: string, versionId2: string): Promise<any> {
    const [version1, version2] = await Promise.all([
      this.versionsCollection.findOne({ versionId: versionId1 }),
      this.versionsCollection.findOne({ versionId: versionId2 })
    ]);

    if (!version1 || !version2) {
      throw new Error('One or both versions not found');
    }

    return {
      version1: {
        id: version1.versionId,
        version: version1.version,
        createdAt: version1.createdAt,
        createdBy: version1.createdBy,
        changes: version1.changes
      },
      version2: {
        id: version2.versionId,
        version: version2.version,
        createdAt: version2.createdAt,
        createdBy: version2.createdBy,
        changes: version2.changes
      },
      differences: this.calculateDifferences(version1.changes, version2.changes)
    };
  }

  private calculateDifferences(changes1: ScheduleChange[], changes2: ScheduleChange[]): any[] {
    const differences = [];
    const fields = new Set([
      ...changes1.map(c => c.field),
      ...changes2.map(c => c.field)
    ]);

    for (const field of fields) {
      const change1 = changes1.find(c => c.field === field);
      const change2 = changes2.find(c => c.field === field);

      if (change1?.newValue !== change2?.newValue) {
        differences.push({
          field,
          version1Value: change1?.newValue,
          version2Value: change2?.newValue
        });
      }
    }

    return differences;
  }

  // Lock/unlock schedule for editing
  async lockSchedule(scheduleId: string, userId: string): Promise<boolean> {
    const result = await this.schedulesCollection.updateOne(
      { 
        id: scheduleId, 
        $or: [
          { locked: false },
          { locked: { $exists: false } },
          { lockedBy: userId }
        ]
      },
      {
        $set: {
          locked: true,
          lockedBy: userId,
          lockedAt: new Date()
        }
      }
    );

    return result.modifiedCount > 0;
  }

  async unlockSchedule(scheduleId: string, userId: string): Promise<boolean> {
    const result = await this.schedulesCollection.updateOne(
      { id: scheduleId, lockedBy: userId },
      {
        $set: { locked: false },
        $unset: { lockedBy: '', lockedAt: '' }
      }
    );

    return result.modifiedCount > 0;
  }
}