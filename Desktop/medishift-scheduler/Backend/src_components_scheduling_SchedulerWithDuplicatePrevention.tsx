import React, { useState, useRef, useCallback } from 'react';
import { protectedApi } from '../../services/api-with-duplicate-prevention';
import { requestManager } from '../../services/request-manager';
import toast from 'react-hot-toast';
import { RefreshCw, AlertTriangle } from 'lucide-react';

export const SchedulerWithDuplicatePrevention: React.FC = () => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [lastGeneratedAt, setLastGeneratedAt] = useState<Date | null>(null);
  const generateButtonRef = useRef<HTMLButtonElement>(null);
  
  const handleGenerateSchedule = useCallback(async () => {
    // Check if already generating
    if (isGenerating) {
      toast.error('Schedule generation already in progress');
      return;
    }
    
    // Check if a request is pending
    if (requestManager.isPending('/schedule/monthly/generate', 'POST')) {
      toast.warning('A schedule generation request is already pending');
      return;
    }
    
    // Disable button immediately
    if (generateButtonRef.current) {
      generateButtonRef.current.disabled = true;
    }
    
    setIsGenerating(true);
    
    try {
      const result = await protectedApi.generateMonthlySchedule({
        month: new Date().getMonth(),
        year: new Date().getFullYear(),
        staffingLevel: 'Normal',
        useMLOptimization: true
      });
      
      if (result.success) {
        toast.success('Schedule generated successfully!');
        setLastGeneratedAt(new Date());
      }
    } catch (error: any) {
      if (error.message.includes('Please wait')) {
        toast.error(error.message, {
          duration: 4000,
          icon: '⏱️'
        });
      } else {
        toast.error('Failed to generate schedule: ' + error.message);
      }
    } finally {
      setIsGenerating(false);
      
      // Re-enable button after a delay
      setTimeout(() => {
        if (generateButtonRef.current) {
          generateButtonRef.current.disabled = false;
        }
      }, 2000);
    }
  }, [isGenerating]);
  
  return (
    <div className="p-6">
      <div className="bg-white rounded-lg shadow-md p-4">
        <h2 className="text-xl font-bold mb-4">Schedule Generator</h2>
        
        {/* Status Indicator */}
        {lastGeneratedAt && (
          <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg">
            <p className="text-sm text-green-800">
              Last generated: {lastGeneratedAt.toLocaleTimeString()}
            </p>
          </div>
        )}
        
        {/* Pending Requests Indicator */}
        {requestManager.getPendingCount() > 0 && (
          <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center">
              <AlertTriangle className="h-4 w-4 text-yellow-600 mr-2" />
              <p className="text-sm text-yellow-800">
                {requestManager.getPendingCount()} request(s) in progress
              </p>
            </div>
          </div>
        )}
        
        {/* Generate Button */}
        <button
          ref={generateButtonRef}
          onClick={handleGenerateSchedule}
          disabled={isGenerating}
          className={`
            flex items-center px-4 py-2 rounded-lg font-medium
            ${isGenerating 
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed' 
              : 'bg-blue-600 text-white hover:bg-blue-700'
            }
            transition-all duration-200
          `}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${isGenerating ? 'animate-spin' : ''}`} />
          {isGenerating ? 'Generating...' : 'Generate Schedule'}
        </button>
      </div>
    </div>
  );
};