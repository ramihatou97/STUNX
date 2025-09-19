import express from 'express';
import { ScheduleVersioningService } from '../services/scheduleVersioning';
import { ConflictResolutionQueue } from '../services/conflictResolutionQueue';
import { RealtimeScheduleService } from '../services/realtimeScheduleUpdates';
import logger from '../utils/logger';

const router = express.Router();

// Initialize services (these would be injected in production)
let versioningService: ScheduleVersioningService;
let conflictQueue: ConflictResolutionQueue;
let realtimeService: RealtimeScheduleService;

// Middleware to check authentication
const authenticate = (req: any, res: any, next: any) => {
  // Implementation of authentication
  next();
};

// Get schedule with version info
router.get('/schedules/:scheduleId', authenticate, async (req, res) => {
  try {
    const { scheduleId } = req.params;
    const { includeHistory } = req.query;
    
    // Get schedule
    const schedule = await req.app.locals.db
      .collection('schedules')
      .findOne({ id: scheduleId });

    if (!schedule) {
      return res.status(404).json({ error: 'Schedule not found' });
    }

    // Include version history if requested
    if (includeHistory === 'true') {
      const history = await versioningService.getVersionHistory(scheduleId);
      schedule.versionHistory = history;
    }

    // Check if locked
    const lockStatus = await realtimeService.isScheduleLocked(scheduleId);
    schedule.lockStatus = lockStatus;

    res.json(schedule);
  } catch (error) {
    logger.error('Failed to get schedule', { error: error.message });
    res.status(500).json({ error: 'Failed to retrieve schedule' });
  }
});

// Update schedule with conflict detection
router.put('/schedules/:scheduleId', authenticate, async (req, res) => {
  try {
    const { scheduleId } = req.params;
    const { changes } = req.body;
    const userId = req.user.id;

    // Check for lock
    const lockStatus = await realtimeService.isScheduleLocked(scheduleId);
    if (lockStatus.locked && lockStatus.lockedBy !== userId) {
      return res.status(409).json({ 
        error: 'Schedule is locked by another user',
        lockedBy: lockStatus.lockedBy
      });
    }

    // Create new version
    const version = await versioningService.createVersion(
      scheduleId,
      changes,
      userId
    );

    // Notify real-time subscribers
    await realtimeService.notifyScheduleChange(
      scheduleId,
      'updated',
      changes,
      userId
    );

    res.json({ 
      success: true, 
      versionId: version.versionId,
      version: version.version
    });
  } catch (error) {
    logger.error('Failed to update schedule', { error: error.message });
    
    // Check if it's a conflict
    if (error.message.includes('conflict')) {
      // Add to conflict resolution queue
      const conflictId = await conflictQueue.addConflict({
        scheduleId: req.params.scheduleId,
        type: 'version_conflict',
        severity: 'medium',
        conflictingChanges: [{
          userId: req.user.id,
          change: req.body.changes,
          timestamp: new Date()
        }],
        metadata: {}
      });

      return res.status(409).json({ 
        error: 'Conflict detected',
        conflictId 
      });
    }

    res.status(500).json({ error: 'Failed to update schedule' });
  }
});

// Get version history
router.get('/schedules/:scheduleId/versions', authenticate, async (req, res) => {
  try {
    const { scheduleId } = req.params;
    const { limit = 10 } = req.query;
    
    const history = await versioningService.getVersionHistory(
      scheduleId, 
      parseInt(limit as string)
    );

    res.json({ history });
  } catch (error) {
    logger.error('Failed to get version history', { error: error.message });
    res.status(500).json({ error: 'Failed to retrieve version history' });
  }
});

// Compare versions
router.get('/schedules/:scheduleId/versions/compare', authenticate, async (req, res) => {
  try {
    const { version1, version2 } = req.query;
    
    if (!version1 || !version2) {
      return res.status(400).json({ 
        error: 'Both version1 and version2 parameters are required' 
      });
    }

    const comparison = await versioningService.compareVersions(
      version1 as string,
      version2 as string
    );

    res.json(comparison);
  } catch (error) {
    logger.error('Failed to compare versions', { error: error.message });
    res.status(500).json({ error: 'Failed to compare versions' });
  }
});

// Rollback to version
router.post('/schedules/:scheduleId/rollback', authenticate, async (req, res) => {
  try {
    const { scheduleId } = req.params;
    const { versionId } = req.body;
    const userId = req.user.id;

    const schedule = await versioningService.rollbackToVersion(
      scheduleId,
      versionId,
      userId
    );

    // Notify subscribers
    await realtimeService.notifyScheduleChange(
      scheduleId,
      'updated',
      { rollback: true, versionId },
      userId
    );

    res.json({ 
      success: true, 
      schedule 
    });
  } catch (error) {
    logger.error('Failed to rollback schedule', { error: error.message });
    res.status(500).json({ error: 'Failed to rollback schedule' });
  }
});

// Lock schedule
router.post('/schedules/:scheduleId/lock', authenticate, async (req, res) => {
  try {
    const { scheduleId } = req.params;
    const userId = req.user.id;

    const locked = await versioningService.lockSchedule(scheduleId, userId);

    if (!locked) {
      return res.status(409).json({ 
        error: 'Schedule is already locked' 
      });
    }

    res.json({ 
      success: true, 
      lockedBy: userId 
    });
  } catch (error) {
    logger.error('Failed to lock schedule', { error: error.message });
    res.status(500).json({ error: 'Failed to lock schedule' });
  }
});

// Unlock schedule
router.post('/schedules/:scheduleId/unlock', authenticate, async (req, res) => {
  try {
    const { scheduleId } = req.params;
    const userId = req.user.id;

    const unlocked = await versioningService.unlockSchedule(scheduleId, userId);

    if (!unlocked) {
      return res.status(403).json({ 
        error: 'You do not hold the lock for this schedule' 
      });
    }

    res.json({ success: true });
  } catch (error) {
    logger.error('Failed to unlock schedule', { error: error.message });
    res.status(500).json({ error: 'Failed to unlock schedule' });
  }
});

// Get conflicts
router.get('/conflicts', authenticate, async (req, res) => {
  try {
    const { status = 'pending', limit = 10 } = req.query;
    
    const conflicts = await req.app.locals.db
      .collection('schedule_conflicts')
      .find({ status })
      .limit(parseInt(limit as string))
      .toArray();

    res.json({ conflicts });
  } catch (error) {
    logger.error('Failed to get conflicts', { error: error.message });
    res.status(500).json({ error: 'Failed to retrieve conflicts' });
  }
});

// Get conflict status
router.get('/conflicts/:conflictId', authenticate, async (req, res) => {
  try {
    const { conflictId } = req.params;
    
    const conflict = await conflictQueue.getConflictStatus(conflictId);

    if (!conflict) {
      return res.status(404).json({ error: 'Conflict not found' });
    }

    res.json(conflict);
  } catch (error) {
    logger.error('Failed to get conflict status', { error: error.message });
    res.status(500).json({ error: 'Failed to retrieve conflict status' });
  }
});

// Manually resolve conflict
router.post('/conflicts/:conflictId/resolve', authenticate, async (req, res) => {
  try {
    const { conflictId } = req.params;
    const { resolution } = req.body;
    const userId = req.user.id;

    // Update conflict with manual resolution
    await req.app.locals.db
      .collection('schedule_conflicts')
      .updateOne(
        { id: conflictId },
        {
          $set: {
            status: 'resolved',
            resolution: {
              ...resolution,
              strategy: 'manual_merge',
              resolvedBy: userId,
              resolvedAt: new Date()
            }
          }
        }
      );

    res.json({ success: true });
  } catch (error) {
    logger.error('Failed to resolve conflict', { error: error.message });
    res.status(500).json({ error: 'Failed to resolve conflict' });
  }
});

export default router;