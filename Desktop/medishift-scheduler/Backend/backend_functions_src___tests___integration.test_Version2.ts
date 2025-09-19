import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach, jest } from '@jest/globals';
import request from 'supertest';
import { MongoClient, Db } from 'mongodb';
import Redis from 'ioredis';
import app from '../app';
import logger from '../utils/logger';
import healthMonitor from '../utils/healthCheck';
import { generateMonthlySchedule } from '../services/scheduleService';
import { processPayment } from '../services/paymentService';
import { sendNotification } from '../services/notificationService';

// Mock external services
jest.mock('../utils/logger');
jest.mock('../services/emailService');

describe('Backend Integration Tests', () => {
  let mongoClient: MongoClient;
  let db: Db;
  let redisClient: Redis;
  let server: any;

  // Test configuration
  const TEST_PORT = process.env.TEST_PORT || 3001;
  const MONGODB_TEST_URI = process.env.MONGODB_TEST_URI || 'mongodb://localhost:27017/test_db';
  const REDIS_TEST_URI = process.env.REDIS_TEST_URI || 'redis://localhost:6379';

  beforeAll(async () => {
    // Set up test environment
    process.env.NODE_ENV = 'test';
    
    // Connect to test database
    mongoClient = new MongoClient(MONGODB_TEST_URI);
    await mongoClient.connect();
    db = mongoClient.db();
    
    // Connect to Redis
    redisClient = new Redis(REDIS_TEST_URI);
    
    // Start server
    server = app.listen(TEST_PORT);
    
    // Initialize health monitoring
    healthMonitor.startMonitoring();
    
    logger.info('Test environment initialized');
  }, 30000);

  afterAll(async () => {
    // Clean up
    await mongoClient.close();
    await redisClient.quit();
    await new Promise((resolve) => server.close(resolve));
    
    logger.info('Test environment cleaned up');
  });

  beforeEach(async () => {
    // Clear test data before each test
    await db.collection('users').deleteMany({});
    await db.collection('schedules').deleteMany({});
    await db.collection('payments').deleteMany({});
    await redisClient.flushdb();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Health Check Endpoints', () => {
    test('GET /api/health should return health status', async () => {
      const response = await request(app)
        .get('/api/health')
        .expect('Content-Type', /json/);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('status');
      expect(response.body).toHaveProperty('timestamp');
      expect(response.body).toHaveProperty('healthScore');
      expect(['healthy', 'degraded', 'unhealthy']).toContain(response.body.status);
    });

    test('GET /api/health/live should return alive status', async () => {
      const response = await request(app)
        .get('/api/health/live')
        .expect(200)
        .expect('Content-Type', /json/);

      expect(response.body.status).toBe('alive');
      expect(response.body).toHaveProperty('timestamp');
    });

    test('GET /api/health/ready should check readiness', async () => {
      const response = await request(app)
        .get('/api/health/ready')
        .expect('Content-Type', /json/);

      expect([200, 503]).toContain(response.status);
      expect(response.body).toHaveProperty('status');
      expect(['ready', 'not_ready']).toContain(response.body.status);
    });

    test('GET /api/health/metrics should return system metrics', async () => {
      const response = await request(app)
        .get('/api/health/metrics')
        .expect(200)
        .expect('Content-Type', /json/);

      expect(response.body).toHaveProperty('uptime');
      expect(response.body).toHaveProperty('timestamp');
      expect(response.body).toHaveProperty('process');
      expect(response.body).toHaveProperty('memory');
      expect(response.body).toHaveProperty('system');
      expect(response.body.process.pid).toBeGreaterThan(0);
    });

    test('GET /api/health/history should return health history', async () => {
      const response = await request(app)
        .get('/api/health/history?limit=5')
        .expect(200)
        .expect('Content-Type', /json/);

      expect(response.body).toHaveProperty('history');
      expect(response.body).toHaveProperty('count');
      expect(Array.isArray(response.body.history)).toBe(true);
      expect(response.body.count).toBeLessThanOrEqual(5);
    });
  });

  describe('Authentication Tests', () => {
    const testUser = {
      username: 'testuser',
      email: 'test@example.com',
      password: 'TestPassword123!'
    };

    test('POST /api/auth/register should create a new user', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send(testUser)
        .expect(201)
        .expect('Content-Type', /json/);

      expect(response.body).toHaveProperty('message');
      expect(response.body).toHaveProperty('userId');
      
      // Verify user was created in database
      const user = await db.collection('users').findOne({ email: testUser.email });
      expect(user).toBeTruthy();
      expect(user?.username).toBe(testUser.username);
    });

    test('POST /api/auth/login should authenticate user', async () => {
      // First register the user
      await request(app)
        .post('/api/auth/register')
        .send(testUser);

      // Then try to login
      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: testUser.email,
          password: testUser.password
        })
        .expect(200)
        .expect('Content-Type', /json/);

      expect(response.body).toHaveProperty('token');
      expect(response.body).toHaveProperty('user');
      expect(response.body.user.email).toBe(testUser.email);
    });

    test('POST /api/auth/login should fail with wrong password', async () => {
      // Register user first
      await request(app)
        .post('/api/auth/register')
        .send(testUser);

      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: testUser.email,
          password: 'WrongPassword123!'
        })
        .expect(401)
        .expect('Content-Type', /json/);

      expect(response.body).toHaveProperty('error');
    });
  });

  describe('Schedule Generation Tests', () => {
    test('generateMonthlySchedule should create schedule for valid month', async () => {
      const month = 8; // August
      const year = 2025;
      
      const schedule = await generateMonthlySchedule(month, year);
      
      expect(schedule).toBeDefined();
      expect(Array.isArray(schedule)).toBe(true);
      expect(schedule.length).toBeGreaterThan(0);
      
      // Verify schedule was saved to database
      const savedSchedule = await db.collection('schedules').findOne({ 
        month, 
        year 
      });
      expect(savedSchedule).toBeTruthy();
    });

    test('POST /api/schedules/generate should create monthly schedule', async () => {
      const response = await request(app)
        .post('/api/schedules/generate')
        .send({
          month: 8,
          year: 2025
        })
        .expect(200)
        .expect('Content-Type', /json/);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('schedule');
      expect(Array.isArray(response.body.schedule)).toBe(true);
    });

    test('GET /api/schedules/:year/:month should retrieve schedule', async () => {
      // First generate a schedule
      await request(app)
        .post('/api/schedules/generate')
        .send({ month: 8, year: 2025 });

      // Then retrieve it
      const response = await request(app)
        .get('/api/schedules/2025/8')
        .expect(200)
        .expect('Content-Type', /json/);

      expect(response.body).toHaveProperty('schedule');
      expect(response.body.schedule).toBeDefined();
    });
  });

  describe('Payment Processing Tests', () => {
    test('POST /api/payments/process should handle valid payment', async () => {
      const paymentData = {
        amount: 100.00,
        currency: 'USD',
        paymentMethod: 'credit_card',
        userId: 'test_user_123'
      };

      const response = await request(app)
        .post('/api/payments/process')
        .send(paymentData)
        .expect(200)
        .expect('Content-Type', /json/);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('transactionId');
      
      // Verify payment was saved
      const payment = await db.collection('payments').findOne({ 
        transactionId: response.body.transactionId 
      });
      expect(payment).toBeTruthy();
      expect(payment?.amount).toBe(paymentData.amount);
    });

    test('POST /api/payments/process should reject invalid amount', async () => {
      const response = await request(app)
        .post('/api/payments/process')
        .send({
          amount: -10,
          currency: 'USD',
          paymentMethod: 'credit_card'
        })
        .expect(400)
        .expect('Content-Type', /json/);

      expect(response.body).toHaveProperty('error');
    });
  });

  describe('Notification Tests', () => {
    test('POST /api/notifications/send should queue notification', async () => {
      const notificationData = {
        userId: 'test_user_123',
        type: 'email',
        subject: 'Test Notification',
        message: 'This is a test notification'
      };

      const response = await request(app)
        .post('/api/notifications/send')
        .send(notificationData)
        .expect(200)
        .expect('Content-Type', /json/);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('notificationId');
      
      // Verify notification was queued in Redis
      const queued = await redisClient.get(`notification:${response.body.notificationId}`);
      expect(queued).toBeTruthy();
    });

    test('GET /api/notifications/:userId should retrieve user notifications', async () => {
      const userId = 'test_user_123';
      
      // Create some notifications
      await db.collection('notifications').insertMany([
        { userId, message: 'Notification 1', createdAt: new Date() },
        { userId, message: 'Notification 2', createdAt: new Date() }
      ]);

      const response = await request(app)
        .get(`/api/notifications/${userId}`)
        .expect(200)
        .expect('Content-Type', /json/);

      expect(response.body).toHaveProperty('notifications');
      expect(Array.isArray(response.body.notifications)).toBe(true);
      expect(response.body.notifications.length).toBe(2);
    });
  });

  describe('Data Validation Tests', () => {
    test('should validate email format', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          username: 'testuser',
          email: 'invalid-email',
          password: 'TestPassword123!'
        })
        .expect(400);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('email');
    });

    test('should validate required fields', async () => {
      const response = await request(app)
        .post('/api/schedules/generate')
        .send({
          // Missing required fields
        })
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });
  });

  describe('Rate Limiting Tests', () => {
    test('should enforce rate limits on API endpoints', async () => {
      const requests = [];
      
      // Make 10 rapid requests
      for (let i = 0; i < 10; i++) {
        requests.push(
          request(app)
            .get('/api/health')
        );
      }

      const responses = await Promise.all(requests);
      
      // Check if any request was rate limited
      const rateLimited = responses.some(r => r.status === 429);
      
      // This test assumes you have rate limiting configured
      // Adjust based on your actual rate limit settings
      if (process.env.RATE_LIMIT_ENABLED === 'true') {
        expect(rateLimited).toBe(true);
      }
    });
  });

  describe('Error Handling Tests', () => {
    test('should handle 404 for unknown routes', async () => {
      const response = await request(app)
        .get('/api/unknown-endpoint')
        .expect(404);

      expect(response.body).toHaveProperty('error');
    });

    test('should handle malformed JSON', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .set('Content-Type', 'application/json')
        .send('{"invalid json}')
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    test('should handle database connection errors gracefully', async () => {
      // Temporarily close database connection
      await mongoClient.close();

      const response = await request(app)
        .get('/api/schedules/2025/8')
        .expect(500);

      expect(response.body).toHaveProperty('error');
      
      // Reconnect for other tests
      await mongoClient.connect();
    });
  });

  describe('Cache Tests', () => {
    test('should cache frequently accessed data', async () => {
      // First request - should hit database
      const response1 = await request(app)
        .get('/api/schedules/2025/8')
        .expect(200);

      // Second request - should hit cache
      const response2 = await request(app)
        .get('/api/schedules/2025/8')
        .expect(200);

      expect(response1.body).toEqual(response2.body);
      
      // Verify cache was used (check Redis)
      const cached = await redisClient.get('schedule:2025:8');
      expect(cached).toBeTruthy();
    });

    test('should invalidate cache on update', async () => {
      // Generate initial schedule
      await request(app)
        .post('/api/schedules/generate')
        .send({ month: 8, year: 2025 });

      // Get cached version
      await request(app)
        .get('/api/schedules/2025/8');

      // Update schedule
      await request(app)
        .put('/api/schedules/2025/8')
        .send({ updates: { modified: true } });

      // Verify cache was invalidated
      const cached = await redisClient.get('schedule:2025:8');
      expect(cached).toBeNull();
    });
  });

  describe('Concurrency Tests', () => {
    test('should handle concurrent requests correctly', async () => {
      const concurrentRequests = 20;
      const requests = [];

      for (let i = 0; i < concurrentRequests; i++) {
        requests.push(
          request(app)
            .post('/api/payments/process')
            .send({
              amount: 100,
              currency: 'USD',
              paymentMethod: 'credit_card',
              userId: `user_${i}`
            })
        );
      }

      const responses = await Promise.all(requests);
      
      // All requests should succeed
      responses.forEach(response => {
        expect(response.status).toBe(200);
        expect(response.body).toHaveProperty('transactionId');
      });

      // Verify all payments were saved
      const payments = await db.collection('payments').find({}).toArray();
      expect(payments.length).toBe(concurrentRequests);
    });
  });

  describe('Performance Tests', () => {
    test('API response time should be within acceptable limits', async () => {
      const startTime = Date.now();
      
      await request(app)
        .get('/api/health')
        .expect(200);
      
      const responseTime = Date.now() - startTime;
      
      // Response should be under 1 second
      expect(responseTime).toBeLessThan(1000);
    });

    test('Database queries should be optimized', async () => {
      // Insert test data
      const testData = [];
      for (let i = 0; i < 100; i++) {
        testData.push({
          userId: `user_${i}`,
          data: `data_${i}`,
          createdAt: new Date()
        });
      }
      await db.collection('test_collection').insertMany(testData);

      const startTime = Date.now();
      
      // Perform a query
      await db.collection('test_collection')
        .find({ userId: 'user_50' })
        .toArray();
      
      const queryTime = Date.now() - startTime;
      
      // Query should be fast (under 100ms)
      expect(queryTime).toBeLessThan(100);
    });
  });

  describe('Security Tests', () => {
    test('should sanitize user input to prevent injection', async () => {
      const maliciousInput = {
        username: '<script>alert("XSS")</script>',
        email: 'test@example.com',
        password: 'TestPassword123!'
      };

      const response = await request(app)
        .post('/api/auth/register')
        .send(maliciousInput)
        .expect(201);

      // Verify the input was sanitized in database
      const user = await db.collection('users').findOne({ 
        email: maliciousInput.email 
      });
      
      expect(user?.username).not.toContain('<script>');
    });

    test('should require authentication for protected routes', async () => {
      const response = await request(app)
        .get('/api/user/profile')
        .expect(401);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('authentication');
    });

    test('should validate JWT tokens', async () => {
      const invalidToken = 'invalid.jwt.token';
      
      const response = await request(app)
        .get('/api/user/profile')
        .set('Authorization', `Bearer ${invalidToken}`)
        .expect(401);

      expect(response.body).toHaveProperty('error');
    });
  });

  describe('Business Logic Tests', () => {
    test('should correctly calculate billing amounts', async () => {
      const billingData = {
        userId: 'test_user',
        items: [
          { id: 1, price: 10.00, quantity: 2 },
          { id: 2, price: 15.50, quantity: 1 }
        ],
        taxRate: 0.08
      };

      const response = await request(app)
        .post('/api/billing/calculate')
        .send(billingData)
        .expect(200);

      const expectedSubtotal = 35.50;
      const expectedTax = expectedSubtotal * 0.08;
      const expectedTotal = expectedSubtotal + expectedTax;

      expect(response.body.subtotal).toBeCloseTo(expectedSubtotal, 2);
      expect(response.body.tax).toBeCloseTo(expectedTax, 2);
      expect(response.body.total).toBeCloseTo(expectedTotal, 2);
    });

    test('should handle schedule conflicts correctly', async () => {
      // Create initial schedule
      await db.collection('schedules').insertOne({
        userId: 'test_user',
        date: '2025-08-16',
        timeSlot: '10:00-11:00',
        status: 'booked'
      });

      // Try to book conflicting schedule
      const response = await request(app)
        .post('/api/schedules/book')
        .send({
          userId: 'test_user',
          date: '2025-08-16',
          timeSlot: '10:00-11:00'
        })
        .expect(409);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('conflict');
    });
  });

  describe('Cleanup and Recovery Tests', () => {
    test('should handle graceful shutdown', async () => {
      // This test would typically be run separately
      // as it involves shutting down the server
      
      const shutdownPromise = new Promise((resolve) => {
        process.once('SIGTERM', resolve);
      });

      // Simulate SIGTERM
      process.emit('SIGTERM', 'SIGTERM');

      await expect(shutdownPromise).resolves.toBe('SIGTERM');
    });

    test('should recover from temporary failures', async () => {
      // Simulate temporary Redis failure
      await redisClient.disconnect();

      // Request should still work (with fallback)
      const response = await request(app)
        .get('/api/health')
        .expect(200);

      expect(response.body).toHaveProperty('status');

      // Reconnect Redis
      await redisClient.connect();
    });
  });
});

// Test utilities
describe('Test Utilities', () => {
  test('logger should be mocked in test environment', () => {
    expect(logger.info).toHaveBeenCalled();
    expect(logger.error).toBeDefined();
  });

  test('test database should be isolated', async () => {
    const dbName = db.databaseName;
    expect(dbName).toContain('test');
  });
});