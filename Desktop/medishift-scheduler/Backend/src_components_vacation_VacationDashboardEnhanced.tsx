import React from 'react';
import { usePreventDuplicates } from '../../hooks/usePreventDuplicates';
import { protectedApi } from '../../services/api-with-duplicate-prevention';
import toast from 'react-hot-toast';

export const VacationDashboardEnhanced: React.FC = () => {
  // Prevent duplicate approval requests
  const [approveRequest, isApproving, approveError] = usePreventDuplicates(
    async (requestId: string) => {
      const result = await protectedApi.approveLeaveRequest(requestId);
      toast.success('Leave request approved');
      return result;
    },
    { minInterval: 5000, showToast: true }
  );
  
  // Prevent duplicate denial requests
  const [denyRequest, isDenying, denyError] = usePreventDuplicates(
    async (requestId: string, justification: string) => {
      const result = await protectedApi.denyLeaveRequest(requestId, justification);
      toast.success('Leave request denied');
      return result;
    },
    { minInterval: 5000, showToast: true }
  );
  
  const handleApprove = async (requestId: string) => {
    try {
      await approveRequest(requestId);
      // Refresh data
    } catch (error) {
      // Error already handled by hook
    }
  };
  
  const handleDeny = async (requestId: string, justification: string) => {
    try {
      await denyRequest(requestId, justification);
      // Refresh data
    } catch (error) {
      // Error already handled by hook
    }
  };
  
  return (
    <div className="p-6">
      {/* Your UI components */}
      <button
        onClick={() => handleApprove('request-123')}
        disabled={isApproving}
        className={`px-4 py-2 rounded ${
          isApproving 
            ? 'bg-gray-300 cursor-not-allowed' 
            : 'bg-green-600 text-white hover:bg-green-700'
        }`}
      >
        {isApproving ? 'Processing...' : 'Approve'}
      </button>
    </div>
  );
};