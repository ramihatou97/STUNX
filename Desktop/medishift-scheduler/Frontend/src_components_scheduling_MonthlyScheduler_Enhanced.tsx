import React, { useState, useEffect, useCallback } from 'react';
import { 
  Calendar, 
  Moon, 
  Sun, 
  AlertTriangle, 
  RefreshCw,
  Download,
  ChevronLeft,
  ChevronRight,
  Info,
  Clock
} from 'lucide-react';
import { 
  MonthlySchedule, 
  CallAssignment, 
  Resident, 
  CrossMonthPostCall 
} from '../../types';
import { useMonthlySchedule } from '../../hooks/useMonthlySchedule';
import { api } from '../../services/api';
import { mlService } from '../../services/ml-service';
import toast from 'react-hot-toast';
import LoadingSpinner from '../common/LoadingSpinner';

interface MonthlySchedulerProps {
  initialMonth?: number;
  initialYear?: number;
}

export const MonthlyScheduler: React.FC<MonthlySchedulerProps> = ({ 
  initialMonth = new Date().getMonth(), 
  initialYear = new Date().getFullYear() 
}) => {
  const [selectedMonth, setSelectedMonth] = useState(initialMonth);
  const [selectedYear, setSelectedYear] = useState(initialYear);
  const [staffingLevel, setStaffingLevel] = useState<'Normal' | 'Shortage'>('Normal');
  const [isGenerating, setIsGenerating] = useState(false);
  const [crossMonthPostCalls, setCrossMonthPostCalls] = useState<CrossMonthPostCall[]>([]);
  const [predictions, setPredictions] = useState<any>(null);
  
  const { 
    schedule, 
    residents, 
    loading, 
    error, 
    refreshSchedule,
    updateAssignment 
  } = useMonthlySchedule(selectedMonth, selectedYear);

  // Load cross-month post-calls from previous month
  useEffect(() => {
    loadCrossMonthPostCalls();
  }, [selectedMonth, selectedYear]);

  // Load AI predictions
  useEffect(() => {
    loadPredictions();
  }, [selectedMonth, selectedYear]);

  const loadCrossMonthPostCalls = async () => {
    try {
      const previousMonth = selectedMonth === 0 ? 11 : selectedMonth - 1;
      const previousYear = selectedMonth === 0 ? selectedYear - 1 : selectedYear;
      
      const result = await api.getCrossMonthPostCalls({
        month: previousMonth,
        year: previousYear,
        nextMonth: selectedMonth,
        nextYear: selectedYear
      });
      
      setCrossMonthPostCalls(result.data.postCalls || []);
    } catch (error) {
      console.error('Failed to load cross-month post-calls:', error);
    }
  };

  const loadPredictions = async () => {
    try {
      const startDate = new Date(selectedYear, selectedMonth, 1);
      const predictions = await mlService.getPredictions(
        startDate.toISOString(), 
        30
      );
      setPredictions(predictions);
    } catch (error) {
      console.error('Failed to load predictions:', error);
    }
  };

  const generateSchedule = async () => {
    setIsGenerating(true);
    try {
      // Include cross-month post-calls in generation
      const result = await api.generateMonthlySchedule({
        month: selectedMonth,
        year: selectedYear,
        staffingLevel,
        crossMonthPostCalls,
        useMLOptimization: true
      });
      
      if (result.data.success) {
        toast.success('Schedule generated successfully!');
        await refreshSchedule();
        
        // Check for anomalies
        const anomalies = await mlService.detectAnomalies(result.data.schedule);
        if (anomalies.hasAnomalies) {
          toast.warning(`${anomalies.count} potential issues detected`, {
            duration: 5000
          });
        }
      }
    } catch (error: any) {
      toast.error(error.message || 'Failed to generate schedule');
    } finally {
      setIsGenerating(false);
    }
  };

  const optimizeWithML = async () => {
    if (!schedule) return;
    
    try {
      const recommendations = await mlService.getRecommendations(
        schedule,
        { staffingLevel, crossMonthPostCalls }
      );
      
      if (recommendations.suggestions.length > 0) {
        // Show recommendations modal
        showRecommendationsModal(recommendations);
      } else {
        toast.success('Schedule is already optimized!');
      }
    } catch (error) {
      toast.error('Failed to get optimization recommendations');
    }
  };

  const showRecommendationsModal = (recommendations: any) => {
    // Implementation for showing ML recommendations
    console.log('ML Recommendations:', recommendations);
  };

  const handleSwapRequest = async (
    fromAssignment: CallAssignment, 
    toDate: Date
  ) => {
    try {
      const result = await api.requestCallSwap({
        fromAssignmentId: fromAssignment.id,
        toDate: toDate.toISOString(),
        reason: 'User requested swap'
      });
      
      if (result.data.success) {
        toast.success('Swap request submitted');
        await refreshSchedule();
      }
    } catch (error) {
      toast.error('Failed to submit swap request');
    }
  };

  const exportSchedule = async () => {
    if (!schedule) return;
    
    try {
      const blob = await api.exportMonthlySchedule({
        scheduleId: schedule.id,
        format: 'excel',
        includeStats: true
      });
      
      const url = window.URL.createObjectURL(blob.data);
      const a = document.createElement('a');
      a.href = url;
      a.download = `schedule_${selectedYear}_${selectedMonth + 1}.xlsx`;
      a.click();
      
      toast.success('Schedule exported successfully!');
    } catch (error) {
      toast.error('Failed to export schedule');
    }
  };

  const renderCalendarDay = (date: Date) => {
    const assignments = schedule?.assignments.filter(a => 
      a.date.toDate().getDate() === date.getDate()
    ) || [];
    
    const crossMonthAssignment = crossMonthPostCalls.find(pc => 
      new Date(pc.date).getDate() === date.getDate()
    );
    
    const isWeekend = date.getDay() === 0 || date.getDay() === 6;
    const isToday = date.toDateString() === new Date().toDateString();
    
    return (
      <div 
        className={`
          min-h-24 p-2 border rounded-lg transition-all cursor-pointer
          ${isWeekend ? 'bg-gray-50' : 'bg-white'}
          ${isToday ? 'ring-2 ring-blue-500' : ''}
          hover:shadow-md
        `}
      >
        <div className="flex justify-between items-start mb-1">
          <span className={`font-bold text-sm ${isWeekend ? 'text-gray-600' : ''}`}>
            {date.getDate()}
          </span>
          {predictions?.highRiskDates?.includes(date.toISOString()) && (
            <AlertTriangle className="h-4 w-4 text-yellow-500" title="High coverage risk predicted" />
          )}
        </div>
        
        <div className="space-y-1">
          {/* Regular assignments */}
          {assignments.map((assignment, idx) => (
            <div 
              key={idx} 
              className={`text-xs px-1 py-0.5 rounded text-white ${
                assignment.type === 'Night' ? 'bg-indigo-600' : 
                assignment.type === 'Weekend' ? 'bg-red-500' : 
                assignment.type === 'Holiday' ? 'bg-purple-500' :
                'bg-blue-500'
              }`}
              title={`${assignment.residentName} - ${assignment.type}`}
            >
              {assignment.type[0]}
            </div>
          ))}
          
          {/* Cross-month post-call */}
          {crossMonthAssignment && (
            <div className="text-xs px-1 py-0.5 rounded bg-orange-500 text-white">
              <Clock className="inline h-3 w-3" /> Post
            </div>
          )}
        </div>
        
        {assignments.length === 0 && !crossMonthAssignment && (
          <span className="text-xs text-gray-400">Unassigned</span>
        )}
      </div>
    );
  };

  if (loading) return <LoadingSpinner />;
  if (error) return <div className="text-red-500">Error: {error}</div>;

  return (
    <div className="p-6 bg-gray-100">
      <div className="bg-white p-6 rounded-xl shadow-lg">
        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold text-gray-800 flex items-center">
            <Calendar className="mr-2 text-indigo-600" />
            Monthly Call Scheduler
          </h1>
          
          {/* Month Navigation */}
          <div className="flex items-center gap-4">
            <button 
              onClick={() => {
                if (selectedMonth === 0) {
                  setSelectedMonth(11);
                  setSelectedYear(y => y - 1);
                } else {
                  setSelectedMonth(m => m - 1);
                }
              }}
              className="p-2 hover:bg-gray-100 rounded"
            >
              <ChevronLeft className="h-5 w-5" />
            </button>
            
            <h2 className="text-lg font-semibold">
              {new Date(selectedYear, selectedMonth).toLocaleDateString('en-US', { 
                month: 'long', 
                year: 'numeric' 
              })}
            </h2>
            
            <button 
              onClick={() => {
                if (selectedMonth === 11) {
                  setSelectedMonth(0);
                  setSelectedYear(y => y + 1);
                } else {
                  setSelectedMonth(m => m + 1);
                }
              }}
              className="p-2 hover:bg-gray-100 rounded"
            >
              <ChevronRight className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap gap-3 mb-6 p-4 bg-gray-50 rounded-lg">
          <select 
            value={staffingLevel}
            onChange={(e) => setStaffingLevel(e.target.value as 'Normal' | 'Shortage')}
            className="px-3 py-2 border rounded-lg"
          >
            <option value="Normal">Normal Staffing</option>
            <option value="Shortage">Staff Shortage</option>
          </select>
          
          <button 
            onClick={generateSchedule} 
            disabled={isGenerating}
            className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
          >
            {isGenerating ? <LoadingSpinner size="sm" /> : <RefreshCw className="mr-2 h-4 w-4"/>}
            Generate
          </button>
          
          <button 
            onClick={optimizeWithML}
            className="flex items-center px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
          >
            Optimize with AI
          </button>
          
          <button 
            onClick={exportSchedule}
            className="flex items-center px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
          >
            <Download className="mr-2 h-4 w-4"/>
            Export
          </button>
        </div>

        {/* Cross-Month Alert */}
        {crossMonthPostCalls.length > 0 && (
          <div className="mb-4 p-3 bg-orange-50 border border-orange-200 rounded-lg">
            <div className="flex items-center">
              <Info className="h-5 w-5 text-orange-600 mr-2" />
              <span className="text-sm text-orange-800">
                {crossMonthPostCalls.length} post-call day(s) carried over from previous month
              </span>
            </div>
          </div>
        )}

        {/* Calendar Grid */}
        <div className="grid grid-cols-7 gap-1">
          {/* Day headers */}
          {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
            <div key={day} className="text-center font-bold text-sm text-gray-600 p-2">
              {day}
            </div>
          ))}
          
          {/* Calendar days */}
          {(() => {
            const daysInMonth = new Date(selectedYear, selectedMonth + 1, 0).getDate();
            const firstDay = new Date(selectedYear, selectedMonth, 1).getDay();
            const days = [];
            
            // Empty cells for alignment
            for (let i = 0; i < firstDay; i++) {
              days.push(<div key={`empty-${i}`} />);
            }
            
            // Actual days
            for (let day = 1; day <= daysInMonth; day++) {
              const date = new Date(selectedYear, selectedMonth, day);
              days.push(
                <div key={day}>
                  {renderCalendarDay(date)}
                </div>
              );
            }
            
            return days;
          })()}
        </div>

        {/* Statistics Panel */}
        {schedule && (
          <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h4 className="font-semibold text-sm text-blue-800">Total Calls</h4>
              <p className="text-2xl font-bold text-blue-600">
                {schedule.assignments.length}
              </p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <h4 className="font-semibold text-sm text-green-800">Coverage Rate</h4>
              <p className="text-2xl font-bold text-green-600">
                {Math.round((schedule.assignments.length / 30) * 100)}%
              </p>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg">
              <h4 className="font-semibold text-sm text-purple-800">Fairness Index</h4>
              <p className="text-2xl font-bold text-purple-600">
                {schedule.metadata.fairnessIndex.toFixed(2)}
              </p>
            </div>
            <div className="bg-orange-50 p-4 rounded-lg">
              <h4 className="font-semibold text-sm text-orange-800">Avg Calls/Resident</h4>
              <p className="text-2xl font-bold text-orange-600">
                {schedule.metadata.averageCallsPerResident.toFixed(1)}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};