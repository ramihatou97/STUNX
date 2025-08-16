import { Server as SocketIOServer } from 'socket.io';
import { Server as HTTPServer } from 'http';
import jwt from 'jsonwebtoken';
import Redis from 'ioredis';
import logger from '../utils/logger';

interface ScheduleUpdate {
  scheduleId: string;
  type: 'created' | 'updated' | 'deleted' | 'locked' | 'unlocked' | 'conflict';
  data: any;
  timestamp: Date;
  userId: string;
  affectedUsers: string[];
}

interface SubscriptionOptions {
  scheduleIds?: string[];
  eventTypes?: string[];
  userId?: string;
}

export class RealtimeScheduleService {
  private io: SocketIOServer;
  private redis: Redis;
  private pubClient: Redis;
  private subClient: Redis;
  private userSockets: Map<string, Set<string>>;
  private socketSubscriptions: Map<string, SubscriptionOptions>;

  constructor(server: HTTPServer, redisUrl: string) {
    // Initialize Socket.IO with CORS
    this.io = new SocketIOServer(server, {
      cors: {
        origin: process.env.CLIENT_URL || 'http://localhost:3000',
        credentials: true
      },
      transports: ['websocket', 'polling']
    });

    // Initialize Redis for pub/sub
    this.redis = new Redis(redisUrl);
    this.pubClient = new Redis(redisUrl);
    this.subClient = new Redis(redisUrl);

    this.userSockets = new Map();
    this.socketSubscriptions = new Map();

    this.initializeSocketServer();
    this.initializeRedisPubSub();
  }

  private initializeSocketServer() {
    // Authentication middleware
    this.io.use(async (socket, next) => {
      try {
        const token = socket.handshake.auth.token;
        const decoded = jwt.verify(token, process.env.JWT_SECRET!) as any;
        socket.data.userId = decoded.userId;
        socket.data.role = decoded.role;
        next();
      } catch (error) {
        next(new Error('Authentication failed'));
      }
    });

    // Connection handler
    this.io.on('connection', (socket) => {
      const userId = socket.data.userId;
      
      logger.info('User connected to real-time updates', { 
        userId, 
        socketId: socket.id 
      });

      // Track user sockets
      if (!this.userSockets.has(userId)) {
        this.userSockets.set(userId, new Set());
      }
      this.userSockets.get(userId)!.add(socket.id);

      // Join user's personal room
      socket.join(`user:${userId}`);

      // Handle subscriptions
      socket.on('subscribe', (options: SubscriptionOptions) => {
        this.handleSubscription(socket, options);
      });

      socket.on('unsubscribe', (options: SubscriptionOptions) => {
        this.handleUnsubscription(socket, options);
      });

      // Handle schedule operations
      socket.on('schedule:lock', async (scheduleId: string) => {
        await this.handleScheduleLock(socket, scheduleId);
      });

      socket.on('schedule:unlock', async (scheduleId: string) => {
        await this.handleScheduleUnlock(socket, scheduleId);
      });

      socket.on('schedule:update', async (data: any) => {
        await this.handleScheduleUpdate(socket, data);
      });

      // Handle presence
      socket.on('schedule:viewing', (scheduleId: string) => {
        this.handleViewingPresence(socket, scheduleId, true);
      });

      socket.on('schedule:stopViewing', (scheduleId: string) => {
        this.handleViewingPresence(socket, scheduleId, false);
      });

      // Handle disconnection
      socket.on('disconnect', () => {
        this.handleDisconnect(socket);
      });

      // Send initial data
      this.sendInitialData(socket);
    });
  }

  private initializeRedisPubSub() {
    // Subscribe to schedule update channel
    this.subClient.subscribe('schedule:updates');

    this.subClient.on('message', (channel, message) => {
      if (channel === 'schedule:updates') {
        const update = JSON.parse(message) as ScheduleUpdate;
        this.broadcastUpdate(update);
      }
    });
  }

  // Handle subscription to specific schedules or events
  private handleSubscription(socket: any, options: SubscriptionOptions) {
    const socketId = socket.id;
    
    // Store subscription preferences
    this.socketSubscriptions.set(socketId, options);

    // Join rooms for specific schedules
    if (options.scheduleIds) {
      options.scheduleIds.forEach(scheduleId => {
        socket.join(`schedule:${scheduleId}`);
      });
    }

    // Join event type rooms
    if (options.eventTypes) {
      options.eventTypes.forEach(eventType => {
        socket.join(`event:${eventType}`);
      });
    }

    logger.info('Socket subscribed', { 
      socketId, 
      options 
    });

    socket.emit('subscribed', { success: true, options });
  }

  private handleUnsubscription(socket: any, options: SubscriptionOptions) {
    const socketId = socket.id;

    // Leave rooms
    if (options.scheduleIds) {
      options.scheduleIds.forEach(scheduleId => {
        socket.leave(`schedule:${scheduleId}`);
      });
    }

    if (options.eventTypes) {
      options.eventTypes.forEach(eventType => {
        socket.leave(`event:${eventType}`);
      });
    }

    // Update subscription preferences
    const currentSubs = this.socketSubscriptions.get(socketId);
    if (currentSubs) {
      if (options.scheduleIds) {
        currentSubs.scheduleIds = currentSubs.scheduleIds?.filter(
          id => !options.scheduleIds!.includes(id)
        );
      }
      if (options.eventTypes) {
        currentSubs.eventTypes = currentSubs.eventTypes?.filter(
          type => !options.eventTypes!.includes(type)
        );
      }
    }

    socket.emit('unsubscribed', { success: true, options });
  }

  // Handle schedule lock request
  private async handleScheduleLock(socket: any, scheduleId: string) {
    const userId = socket.data.userId;
    
    try {
      // Check if schedule is already locked
      const lockKey = `schedule:lock:${scheduleId}`;
      const existingLock = await this.redis.get(lockKey);

      if (existingLock && existingLock !== userId) {
        socket.emit('schedule:lockFailed', {
          scheduleId,
          reason: 'Already locked by another user',
          lockedBy: existingLock
        });
        return;
      }

      // Acquire lock with 5-minute expiry
      await this.redis.setex(lockKey, 300, userId);

      // Notify all users viewing this schedule
      const update: ScheduleUpdate = {
        scheduleId,
        type: 'locked',
        data: { lockedBy: userId },
        timestamp: new Date(),
        userId,
        affectedUsers: []
      };

      await this.publishUpdate(update);

      socket.emit('schedule:lockSuccess', { scheduleId });
      
      logger.info('Schedule locked', { scheduleId, userId });
    } catch (error) {
      socket.emit('schedule:lockFailed', {
        scheduleId,
        reason: error.message
      });
    }
  }

  // Handle schedule unlock request
  private async handleScheduleUnlock(socket: any, scheduleId: string) {
    const userId = socket.data.userId;
    
    try {
      const lockKey = `schedule:lock:${scheduleId}`;
      const existingLock = await this.redis.get(lockKey);

      if (existingLock !== userId) {
        socket.emit('schedule:unlockFailed', {
          scheduleId,
          reason: 'You do not hold the lock'
        });
        return;
      }

      // Release lock
      await this.redis.del(lockKey);

      // Notify all users
      const update: ScheduleUpdate = {
        scheduleId,
        type: 'unlocked',
        data: { unlockedBy: userId },
        timestamp: new Date(),
        userId,
        affectedUsers: []
      };

      await this.publishUpdate(update);

      socket.emit('schedule:unlockSuccess', { scheduleId });
      
      logger.info('Schedule unlocked', { scheduleId, userId });
    } catch (error) {
      socket.emit('schedule:unlockFailed', {
        scheduleId,
        reason: error.message
      });
    }
  }

  // Handle schedule update
  private async handleScheduleUpdate(socket: any, data: any) {
    const userId = socket.data.userId;
    const { scheduleId, changes } = data;

    try {
      // Verify user has lock
      const lockKey = `schedule:lock:${scheduleId}`;
      const lock = await this.redis.get(lockKey);

      if (lock !== userId) {
        socket.emit('schedule:updateFailed', {
          scheduleId,
          reason: 'Schedule is not locked by you'
        });
        return;
      }

      // Process update (would normally update database here)
      const update: ScheduleUpdate = {
        scheduleId,
        type: 'updated',
        data: changes,
        timestamp: new Date(),
        userId,
        affectedUsers: await this.getAffectedUsers(scheduleId)
      };

      await this.publishUpdate(update);

      socket.emit('schedule:updateSuccess', { scheduleId, changes });
      
      logger.info('Schedule updated', { scheduleId, userId });
    } catch (error) {
      socket.emit('schedule:updateFailed', {
        scheduleId,
        reason: error.message
      });
    }
  }

  // Handle viewing presence
  private handleViewingPresence(socket: any, scheduleId: string, isViewing: boolean) {
    const userId = socket.data.userId;
    const presenceKey = `schedule:viewers:${scheduleId}`;

    if (isViewing) {
      socket.join(`schedule:${scheduleId}:viewers`);
      this.redis.sadd(presenceKey, userId);
      
      // Notify others who's viewing
      socket.to(`schedule:${scheduleId}`).emit('schedule:viewerJoined', {
        scheduleId,
        userId
      });
    } else {
      socket.leave(`schedule:${scheduleId}:viewers`);
      this.redis.srem(presenceKey, userId);
      
      socket.to(`schedule:${scheduleId}`).emit('schedule:viewerLeft', {
        scheduleId,
        userId
      });
    }
  }

  // Handle disconnection
  private handleDisconnect(socket: any) {
    const userId = socket.data.userId;
    const socketId = socket.id;

    // Remove from user sockets
    const userSocketSet = this.userSockets.get(userId);
    if (userSocketSet) {
      userSocketSet.delete(socketId);
      if (userSocketSet.size === 0) {
        this.userSockets.delete(userId);
      }
    }

    // Clean up subscriptions
    this.socketSubscriptions.delete(socketId);

    // Release any locks held by this socket
    this.releaseUserLocks(userId);

    logger.info('User disconnected', { userId, socketId });
  }

  // Send initial data when user connects
  private async sendInitialData(socket: any) {
    const userId = socket.data.userId;

    try {
      // Send user's active schedules
      const activeSchedules = await this.getUserActiveSchedules(userId);
      socket.emit('initial:schedules', activeSchedules);

      // Send any pending conflicts
      const pendingConflicts = await this.getUserPendingConflicts(userId);
      socket.emit('initial:conflicts', pendingConflicts);

      // Send current viewers for subscribed schedules
      const subscriptions = this.socketSubscriptions.get(socket.id);
      if (subscriptions?.scheduleIds) {
        for (const scheduleId of subscriptions.scheduleIds) {
          const viewers = await this.redis.smembers(`schedule:viewers:${scheduleId}`);
          socket.emit('schedule:currentViewers', { scheduleId, viewers });
        }
      }
    } catch (error) {
      logger.error('Failed to send initial data', { userId, error: error.message });
    }
  }

  // Publish update to Redis pub/sub
  async publishUpdate(update: ScheduleUpdate): Promise<void> {
    await this.pubClient.publish('schedule:updates', JSON.stringify(update));
  }

  // Broadcast update to relevant users
  private broadcastUpdate(update: ScheduleUpdate) {
    // Broadcast to schedule room
    this.io.to(`schedule:${update.scheduleId}`).emit('schedule:update', update);

    // Broadcast to event type room
    this.io.to(`event:${update.type}`).emit('schedule:update', update);

    // Broadcast to affected users
    update.affectedUsers.forEach(userId => {
      this.io.to(`user:${userId}`).emit('schedule:update', update);
    });

    // Store update in history
    this.storeUpdateHistory(update);
  }

  // Store update history
  private async storeUpdateHistory(update: ScheduleUpdate) {
    const historyKey = `schedule:history:${update.scheduleId}`;
    
    // Store in Redis with expiry (7 days)
    await this.redis.zadd(
      historyKey, 
      update.timestamp.getTime(), 
      JSON.stringify(update)
    );
    await this.redis.expire(historyKey, 604800);

    // Trim to keep only last 100 updates
    await this.redis.zremrangebyrank(historyKey, 0, -101);
  }

  // Helper methods
  private async getAffectedUsers(scheduleId: string): Promise<string[]> {
    // Implementation to get users affected by schedule change
    // This would query your database
    return [];
  }

  private async getUserActiveSchedules(userId: string): Promise<any[]> {
    // Implementation to get user's active schedules
    return [];
  }

  private async getUserPendingConflicts(userId: string): Promise<any[]> {
    // Implementation to get user's pending conflicts
    return [];
  }

  private async releaseUserLocks(userId: string) {
    // Find and release all locks held by user
    const pattern = 'schedule:lock:*';
    const keys = await this.redis.keys(pattern);
    
    for (const key of keys) {
      const lockHolder = await this.redis.get(key);
      if (lockHolder === userId) {
        await this.redis.del(key);
        
        // Notify about unlock
        const scheduleId = key.replace('schedule:lock:', '');
        const update: ScheduleUpdate = {
          scheduleId,
          type: 'unlocked',
          data: { reason: 'User disconnected' },
          timestamp: new Date(),
          userId,
          affectedUsers: []
        };
        await this.publishUpdate(update);
      }
    }
  }

  // Public methods for external use
  async notifyScheduleChange(scheduleId: string, changeType: string, data: any, userId: string) {
    const update: ScheduleUpdate = {
      scheduleId,
      type: changeType as any,
      data,
      timestamp: new Date(),
      userId,
      affectedUsers: await this.getAffectedUsers(scheduleId)
    };

    await this.publishUpdate(update);
  }

  async notifyConflict(conflict: any) {
    this.io.emit('conflict:detected', conflict);
  }

  // Get current viewers of a schedule
  async getScheduleViewers(scheduleId: string): Promise<string[]> {
    return await this.redis.smembers(`schedule:viewers:${scheduleId}`);
  }

  // Check if schedule is locked
  async isScheduleLocked(scheduleId: string): Promise<{ locked: boolean; lockedBy?: string }> {
    const lockKey = `schedule:lock:${scheduleId}`;
    const lockedBy = await this.redis.get(lockKey);
    
    return {
      locked: !!lockedBy,
      lockedBy: lockedBy || undefined
    };
  }
}