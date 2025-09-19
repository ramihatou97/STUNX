backend/functions/
├── src/
│   ├── index.ts                    # Main exports (FIXED)
│   ├── config/
│   │   └── firebase.ts             # Firebase initialization
│   ├── vacation/
│   │   └── vacation-analyzer.ts    # ACTUAL vacation analyzer
│   ├── auditing/
│   │   └── conflict-detector.ts    # Conflict auditor
│   ├── education/
│   │   └── epaHandler.ts          # EPA assignments
│   ├── notifications/
│   │   └── notification-service.ts # Notifications
│   ├── scheduling/
│   │   ├── yearly-scheduler.ts     # Yearly rotation
│   │   ├── monthly-scheduler.ts    # Monthly calls
│   │   └── weekly-scheduler.ts     # Weekly clinical
│   ├── analytics/
│   │   └── analytics-engine.ts     # Reports & predictions
│   ├── utils/
│   │   ├── error-handler.ts
│   │   ├── validators.ts
│   │   └── date-helpers.ts
│   └── types/
│       └── index.ts                # Local type extensions
├── lib/                            # Compiled JS output
├── package.json
├── tsconfig.json
├── .eslintrc.js
└── jest.config.js