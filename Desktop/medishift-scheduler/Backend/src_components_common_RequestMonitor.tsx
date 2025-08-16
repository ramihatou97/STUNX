import React, { useState, useEffect } from 'react';
import { requestManager } from '../../services/request-manager';
import { Activity } from 'lucide-react';

export const RequestMonitor: React.FC = () => {
  const [pendingCount, setPendingCount] = useState(0);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setPendingCount(requestManager.getPendingCount());
    }, 500);
    
    return () => clearInterval(interval);
  }, []);
  
  if (pendingCount === 0) return null;
  
  return (
    <div className="fixed bottom-4 right-4 bg-blue-600 text-white px-4 py-2 rounded-lg shadow-lg flex items-center">
      <Activity className="h-4 w-4 mr-2 animate-pulse" />
      <span className="text-sm font-medium">
        {pendingCount} request{pendingCount > 1 ? 's' : ''} in progress
      </span>
    </div>
  );
};

// Add to your App.tsx or Layout component
// <RequestMonitor />