/**
 * KOO Platform Main Application - Simplified Single-User Edition
 * React 18 + TypeScript + Material-UI
 * No complex authentication flows - always authenticated
 */

import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, CircularProgress, Box } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

// Core components
import ErrorBoundary from './components/common/ErrorBoundary';
import { AuthProvider } from './contexts/AuthContext';

// Lazy load pages for better performance
const ChapterViewer = lazy(() => import('./components/chapters/ChapterViewer'));
const AcademicDashboard = lazy(() => import('./components/research/AcademicDashboard'));
const EnhancedResearchAnalytics = lazy(() => import('./components/research/EnhancedResearchAnalytics'));

// Simplified theme configuration
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: [
      'Inter',
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      'sans-serif',
    ].join(','),
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        },
      },
    },
  },
});

// React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

// Loading fallback component
const LoadingFallback: React.FC = () => (
  <Box
    display="flex"
    justifyContent="center"
    alignItems="center"
    minHeight="200px"
  >
    <CircularProgress />
  </Box>
);

// Simplified Layout Component
const SimpleLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <Box sx={{
      minHeight: '100vh',
      bgcolor: 'background.default',
      p: 2
    }}>
      <Box sx={{
        maxWidth: '1200px',
        mx: 'auto',
        bgcolor: 'background.paper',
        borderRadius: 2,
        p: 3,
        minHeight: 'calc(100vh - 32px)'
      }}>
        {children}
      </Box>
    </Box>
  );
};

// Simplified application routes (no auth protection needed)
const AppRoutes: React.FC = () => {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <Routes>
        <Route path="/" element={
          <SimpleLayout>
            <ChapterViewer />
          </SimpleLayout>
        } />

        <Route path="/chapters/:id" element={
          <SimpleLayout>
            <ChapterViewer />
          </SimpleLayout>
        } />

        <Route path="/research" element={
          <SimpleLayout>
            <AcademicDashboard />
          </SimpleLayout>
        } />

        <Route path="/research/analytics" element={
          <SimpleLayout>
            <EnhancedResearchAnalytics />
          </SimpleLayout>
        } />

        {/* Catch all route */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  );
};

// Main App component
const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <AuthProvider>
            <Router>
              <AppRoutes />
            </Router>
          </AuthProvider>
          {process.env.NODE_ENV === 'development' && (
            <ReactQueryDevtools initialIsOpen={false} />
          )}
        </ThemeProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
};

export default App;