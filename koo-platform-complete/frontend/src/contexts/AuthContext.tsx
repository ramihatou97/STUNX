/**
 * Simplified Authentication Context for Single-User KOO Platform
 * Always authenticated - no login/logout flows needed
 */

import React, { createContext, useContext, ReactNode } from 'react';

interface User {
  id: number;
  username: string;
  email: string;
  full_name: string;
  role: string;
}

interface AuthContextType {
  user: User;
  isAuthenticated: boolean;
  isLoading: boolean;
  // Simplified - no login/logout methods needed
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Single admin user (always you)
const ADMIN_USER: User = {
  id: 1,
  username: 'admin',
  email: process.env.REACT_APP_ADMIN_EMAIL || 'admin@koo-platform.com',
  full_name: process.env.REACT_APP_ADMIN_NAME || 'Admin User',
  role: 'super_admin'
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  // Always authenticated in single-user mode
  const authValue: AuthContextType = {
    user: ADMIN_USER,
    isAuthenticated: true,
    isLoading: false
  };

  return (
    <AuthContext.Provider value={authValue}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Helper hook for checking permissions (always true in single-user)
export const usePermissions = () => {
  return {
    canRead: true,
    canWrite: true,
    canDelete: true,
    canAdmin: true,
    isOwner: true
  };
};