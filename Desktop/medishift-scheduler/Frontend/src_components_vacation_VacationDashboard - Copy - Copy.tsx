import React, { useState } from 'react';
import { 
  CheckCircle, 
  XCircle, 
  AlertTriangle, 
  Shield, 
  BarChart2,
  Eye,
  MessageSquare
} from 'lucide-react';
import { doc, updateDoc } from 'firebase/firestore';
import { db } from '../../firebase-config';
import { LeaveRequest, LeaveAnalysisReport } from '../../types';
import toast from 'react-hot-toast';

interface VacationDashboardProps {
  requests: LeaveRequest[];
  reports: Record<string, LeaveAnalysisReport>;
  onRefresh: () => void;
}

interface RecommendationBadgeProps {
  recommendation: LeaveAnalysisReport['overallRecommendation'];
}

const RecommendationBadge: React.FC<RecommendationBadgeProps> = ({ recommendation }) => {
  const styles = {
    'Approve': { 
      Icon: CheckCircle, 
      color: 'text-green-500', 
      bgColor: 'bg-green-50',
      label: 'Approve' 
    },
    'Flagged for Review': { 
      Icon: AlertTriangle, 
      color: 'text-yellow-500',
      bgColor: 'bg-yellow-50', 
      label: 'Flagged' 
    },
    'Deny': { 
      Icon: XCircle, 
      color: 'text-red-500',
      bgColor: 'bg-red-50', 
      label: 'Deny' 
    },
  };
  
  const style = styles[recommendation] || styles['Flagged for Review'];
  const { Icon, color, bgColor, label } = style;
  
  return (
    <div className={`flex items-center font-bold px-2 py-1 rounded-lg ${color} ${bgColor}`}>
      <Icon className="h-5 w-5 mr-1" /> 
      {label}
    </div>
  );
};

interface DenialModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (justification: string) => void;
  defaultReason: string;
}

const DenialModal: React.FC<DenialModalProps> = ({ 
  isOpen, 
  onClose, 
  onConfirm, 
  defaultReason 
}) => {
  const [justification, setJustification] = useState(defaultReason);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6">
        <h3 className="text-lg font-bold mb-4">Deny Leave Request</h3>
        <p className="text-sm text-gray-600 mb-4">
          Please provide a detailed justification for this denial.
        </p>
        <textarea
          value={justification}
          onChange={(e) => setJustification(e.target.value)}
          className="w-full p-3 border rounded-lg resize-none focus:ring-2 focus:ring-blue-500"
          rows={4}
          placeholder="Enter justification..."
        />
        <div className="flex justify-end gap-2 mt-4">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg"
          >
            Cancel
          </button>
          <button
            onClick={() => {
              if (justification.trim()) {
                onConfirm(justification);
                onClose();
              } else {
                toast.error('Please provide a justification');
              }
            }}
            className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600"
          >
            Confirm Denial
          </button>
        </div>
      </div>
    </div>
  );
};

export const VacationDashboard: React.FC<VacationDashboardProps> = ({ 
  requests, 
  reports,
  onRefresh 
}) => {
  const [selectedRequest, setSelectedRequest] = useState<LeaveRequest | null>(null);
  const [showDenialModal, setShowDenialModal] = useState(false);
  const [processingId, setProcessingId] = useState<string | null>(null);

  const handleDeny = async (request: LeaveRequest, justification: string) => {
    setProcessingId(request.id);
    try {
      const requestRef = doc(db, 'leaveRequests', request.id);
      await updateDoc(requestRef, {
        status: 'Denied',
        denialJustification: justification,
        updatedAt: new Date()
      });
      
      toast.success('Leave request denied');
      onRefresh();
    } catch (error) {
      toast.error('Failed to deny request');
      console.error(error);
    } finally {
      setProcessingId(null);
    }
  };

  const handleApprove = async (request: LeaveRequest) => {
    setProcessingId(request.id);
    try {
      const requestRef = doc(db, 'leaveRequests', request.id);
      await updateDoc(requestRef, { 
        status: 'Approved',
        updatedAt: new Date()
      });
      
      toast.success('Leave request approved');
      onRefresh();
    } catch (error) {
      toast.error('Failed to approve request');
      console.error(error);
    } finally {
      setProcessingId(null);
    }
  };

  const pendingRequests = requests.filter(r => r.status === 'Pending Approval');

  if (pendingRequests.length === 0) {
    return (
      <div className="bg-gray-50 p-6 rounded-lg">
        <h2 className="text-2xl font-bold mb-4">Vacation & Leave Dashboard</h2>
        <div className="text-center py-8 text-gray-500">
          No pending leave requests at this time.
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-50 p-6 rounded-lg">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Vacation & Leave Dashboard</h2>
        <span className="bg-blue-100 text-blue-700 px-3 py-1 rounded-full text-sm font-medium">
          {pendingRequests.length} Pending
        </span>
      </div>
      
      <div className="space-y-4">
        {pendingRequests.map(request => {
          const report = reports[request.id];
          const isProcessing = processingId === request.id;
          
          if (!report) {
            return (
              <div key={request.id} className="bg-white p-4 rounded-lg shadow-md animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-1/3 mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-1/2"></div>
              </div>
            );
          }

          return (
            <div 
              key={request.id} 
              className={`bg-white p-4 rounded-lg shadow-md border-l-4 transition-all ${
                report.overallRecommendation === 'Deny' ? 'border-red-500' :
                report.overallRecommendation === 'Approve' ? 'border-green-500' :
                'border-yellow-500'
              } ${isProcessing ? 'opacity-50' : ''}`}
            >
              <div className="flex justify-between items-start">
                <div>
                  <p className="font-bold text-lg">
                    {request.residentName} 
                    <span className="ml-2 text-sm text-gray-500">
                      (PGY-{request.pgyLevel})
                    </span>
                  </p>
                  <p className="text-sm text-gray-600">
                    {request.type} Request: {' '}
                    {request.startDate.toDate().toLocaleDateString()} - {' '}
                    {request.endDate.toDate().toLocaleDateString()}
                  </p>
                  {request.reason && (
                    <p className="text-sm text-gray-500 mt-1">
                      <MessageSquare className="inline h-3 w-3 mr-1" />
                      {request.reason}
                    </p>
                  )}
                </div>
                <div className="text-right">
                  <RecommendationBadge recommendation={report.overallRecommendation} />
                  <p className="text-xs text-gray-500 mt-1">Analysis Complete</p>
                </div>
              </div>
              
              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 border-t pt-4">
                <div className="bg-gray-100 p-3 rounded">
                  <h4 className="font-semibold flex items-center text-sm">
                    <Shield className="h-4 w-4 mr-2 text-red-600"/>
                    Projected Coverage Risk
                  </h4>
                  <p className={`mt-1 text-lg font-bold ${
                    report.estimatedCoverageImpact.projectedCoverageRisk === 'High' ? 'text-red-600' : 
                    report.estimatedCoverageImpact.projectedCoverageRisk === 'Medium' ? 'text-yellow-600' : 
                    'text-green-600'
                  }`}>
                    {report.estimatedCoverageImpact.projectedCoverageRisk}
                  </p>
                </div>
                <div className="bg-gray-100 p-3 rounded">
                  <h4 className="font-semibold flex items-center text-sm">
                    <BarChart2 className="h-4 w-4 mr-2 text-blue-600"/>
                    Fairness Score
                  </h4>
                  <p className="mt-1 text-sm">
                    Score: <span className="font-bold text-lg">{report.fairnessScore.score}/100</span>
                  </p>
                  <p className="text-xs text-gray-500">
                    Historical Approval: {' '}
                    <span className="font-medium">
                      {(report.fairnessScore.historicalSuccessRateForPeriod * 100).toFixed(0)}%
                    </span>
                  </p>
                </div>
              </div>

              <div className="mt-4 flex justify-end space-x-2">
                <button 
                  onClick={() => setSelectedRequest(request)}
                  className="px-3 py-1 text-sm bg-gray-200 rounded hover:bg-gray-300 flex items-center"
                  disabled={isProcessing}
                >
                  <Eye className="h-3 w-3 mr-1" />
                  View Details
                </button>
                <button 
                  onClick={() => {
                    setSelectedRequest(request);
                    setShowDenialModal(true);
                  }}
                  className="px-3 py-1 text-sm bg-red-500 text-white rounded hover:bg-red-600"
                  disabled={isProcessing}
                >
                  Deny
                </button>
                <button 
                  onClick={() => handleApprove(request)}
                  className="px-3 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600"
                  disabled={isProcessing}
                >
                  Approve
                </button>
              </div>
            </div>
          );
        })}
      </div>
      
      {/* Denial Modal */}
      {selectedRequest && (
        <DenialModal
          isOpen={showDenialModal}
          onClose={() => {
            setShowDenialModal(false);
            setSelectedRequest(null);
          }}
          onConfirm={(justification) => {
            if (selectedRequest) {
              handleDeny(selectedRequest, justification);
            }
          }}
          defaultReason={
            selectedRequest && reports[selectedRequest.id]
              ? `Request denied due to ${reports[selectedRequest.id].estimatedCoverageImpact.projectedCoverageRisk} projected coverage risk.`
              : ''
          }
        />
      )}
    </div>
  );
};