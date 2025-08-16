import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { onAuthStateChanged, User } from 'firebase/auth';
import { auth } from './firebase-config';

// Contexts
import { AuthProvider } from './contexts/AuthContext';
import { NotificationProvider } from './contexts/NotificationContext';
import { ScheduleProvider } from './contexts/ScheduleContext';

// Pages
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Schedule from './pages/Schedule';
import Vacation from './pages/Vacation';
import Analytics from './pages/Analytics';
import Admin from './pages/Admin';
import NotFound from './pages/NotFound';

// Components
import Layout from './components/common/Layout';
import PrivateRoute from './components/auth/PrivateRoute';
import RoleGuard from './components/auth/RoleGuard';
import LoadingSpinner from './components/common/LoadingSpinner';
import ErrorBoundary from './components/common/ErrorBoundary';

// Styles
import './styles/globals.css';

const App: React.FC = () => {
  const [user, setUser] = useState<User | null>(null);
  const [isAdmin, setIsAdmin] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      setUser(user);
      
      if (user) {
        // Check admin status from custom claims
        const idTokenResult = await user.getIdTokenResult();
        setIsAdmin(idTokenResult.claims.admin === true);
      } else {
        setIsAdmin(false);
      }
      
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  if (loading) {
    return <LoadingSpinner fullScreen />;
  }

  return (
    <ErrorBoundary>
      <AuthProvider value={{ user, isAdmin }}>
        <NotificationProvider>
          <ScheduleProvider>
            <Router>
              <Routes>
                {/* Public Routes */}
                <Route path="/login" element={
                  user ? <Navigate to="/" replace /> : <Login />
                } />
                
                {/* Protected Routes */}
                <Route element={<PrivateRoute />}>
                  <Route element={<Layout />}>
                    <Route path="/" element={<Dashboard />} />
                    <Route path="/schedule/*" element={<Schedule />} />
                    <Route path="/vacation/*" element={<Vacation />} />
                    <Route path="/analytics" element={<Analytics />} />
                    
                    {/* Admin Routes */}
                    <Route element={<RoleGuard allowedRoles={['admin']} />}>
                      <Route path="/admin/*" element={<Admin />} />
                    </Route>
                  </Route>
                </Route>
                
                {/* 404 */}
                <Route path="*" element={<NotFound />} />
              </Routes>
            </Router>
          </ScheduleProvider>
        </NotificationProvider>
      </AuthProvider>
    </ErrorBoundary>
  );
};

export default App;