# MediShift: AI-Powered Neurosurgery Residency Management Platform

![Version](https://img.shields.io/badge/version-4.0.0-blue)
![Status](https://img.shields.io/badge/status-production-green)
![License](https://img.shields.io/badge/license-MIT-green)

## 🏥 Overview

MediShift is a comprehensive, AI-powered application designed to revolutionize the management of neurosurgery residency programs. Built with cutting-edge technology and medical education best practices, it handles complex scheduling, education tracking, and program administration with unprecedented efficiency.

## ✨ Core Features

### 📅 **Hierarchical AI Scheduling**
- **Yearly Scheduler**: Manages rotation blocks with PARO compliance
- **Monthly Call Scheduler**: Implements getMaxCalls logic with Normal/Shortage modes
- **Weekly Clinical Scheduler**: Optimizes OR and clinic assignments

### 🏖️ **Intelligent Leave Management**
- **Vacation Analyzer**: AI-powered fairness and coverage analysis
- **Conflict Detector**: Automated nightly audits for schedule conflicts
- **Predictive Planning**: Historical data-driven approval recommendations

### 📚 **Educational Tracking (EPAs)**
- Automatic EPA assignment from OR cases
- Real-time progress tracking
- Competency-based assessment integration

### 📊 **Advanced Analytics**
- Real-time dashboards with predictive metrics
- Fairness distribution analysis (Gini coefficient)
- EPA completion risk predictions
- OR utilization reports

### 🔔 **Smart Notifications**
- Multi-channel alerts (in-app, email, push)
- Priority-based routing
- Automated reminders for EPAs and schedules

## 🛠️ Technology Stack

- **Frontend**: React 18, TypeScript, Material-UI, Redux Toolkit
- **Backend**: Firebase Functions (Node.js 18), TypeScript
- **Database**: Cloud Firestore
- **Authentication**: Firebase Auth
- **Hosting**: Firebase Hosting
- **Analytics**: Custom analytics engine with predictive ML
- **Data Extraction**: Python microservice for document parsing

## 📁 Project Structure

```
medishift/
├── frontend/          # React TypeScript application
├── backend/          # Firebase Cloud Functions
├── shared/           # Shared TypeScript types
├── data-extraction/  # Python document parser
└── docs/            # Documentation
```

## 🚀 Getting Started

### Prerequisites

- Node.js (v18+)
- Firebase CLI (`npm install -g firebase-tools`)
- Python 3.8+ (for data extraction)
- A Firebase project with Blaze plan

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/medishift.git
   cd medishift
   ```

2. **Setup Firebase**
   ```bash
   firebase login
   firebase use neuroman-prod
   ```

3. **Install dependencies**
   ```bash
   # Frontend
   cd frontend && npm install
   
   # Backend
   cd ../backend/functions && npm install
   
   # Python extractor
   cd ../../data-extraction
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy example files
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Development

```bash
# Start Firebase emulators
firebase emulators:start

# Start frontend (new terminal)
cd frontend && npm start

# Run tests
npm test

# Build for production
npm run build
```

### Deployment

```bash
# Deploy everything
firebase deploy

# Deploy specific services
firebase deploy --only functions
firebase deploy --only hosting
firebase deploy --only firestore:rules
```

## 📊 Key Algorithms

### getMaxCalls Logic
The system implements a two-tiered call limit system:
- **Normal Mode**: `min(PARO_cap, PGY_target)`
- **Shortage Mode**: `PARO_cap` only

### Fairness Distribution
Uses Gini coefficient to ensure equitable call distribution across residents.

### Coverage Analysis
Predictive algorithms analyze historical patterns to forecast coverage risks.

## 🔒 Security

- Row-level security with Firestore rules
- Role-based access control (RBAC)
- HIPAA compliance considerations
- Encrypted data transmission
- Regular security audits

## 📈 Performance

- Sub-second schedule generation for 30+ residents
- Real-time conflict detection
- Optimized Firestore queries with composite indexes
- Lazy loading and code splitting
- CDN-cached static assets

## 🧪 Testing

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run specific test suite
npm test -- monthly-scheduler.test.ts
```

## 📝 Documentation

Comprehensive documentation available in `/docs`:
- [API Reference](./docs/api.md)
- [Architecture Guide](./docs/architecture.md)
- [Deployment Guide](./docs/deployment.md)
- [User Manual](./docs/user-manual.md)

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see [LICENSE.md](LICENSE.md) for details.

## 👥 Team

- **Lead Developer**: @ramihatou97
- **Medical Advisors**: Neurosurgery Department
- **UI/UX Design**: Design Team

## 🏆 Acknowledgments

- UHN Neurosurgery Residency Program
- PARO Guidelines Committee
- Firebase Developer Community

## 📞 Support

- **Email**: support@medishift.com
- **Documentation**: https://docs.medishift.com
- **Issues**: GitHub Issues

---

**Version 4.0.0** | Released: August 2025 | © MediShift Team